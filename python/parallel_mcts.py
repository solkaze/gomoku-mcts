"""
並列バッチMCTS

複数の対局を同時に進行させ、各MCTSが「NN推論が必要」になったら
全ゲーム分をまとめて1回のGPU推論で処理する。

設計:
  ・各ゲームは独立したMCTSツリーを持つ
  ・1ステップ = 全ゲームのMCTSを「葉に到達するまで」進める
  ・葉に到達したノードの盤面をまとめてバッチ推論
  ・結果を各ゲームのツリーに展開・backprop
  ・これをn_sim回繰り返す
"""

import math
import numpy as np
import torch
from typing import Optional
from network import GomokuNet, BOARD_SIZE
from mcts import check_winner, get_legal_moves, Node, C_PUCT


class GameMCTS:
    """1ゲーム分のMCTS状態を保持"""

    def __init__(self, board: np.ndarray, to_play: int,
                 dirichlet_alpha: float, dirichlet_eps: float, add_noise: bool):
        self.root = Node(board.copy(), to_play)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise
        self.root_expanded = False

        # 現在のシミュレーションで辿っているパス（葉に到達したら推論待ちになる）
        self.pending_path: Optional[list[Node]] = None
        self.pending_leaf: Optional[Node] = None

    def add_dirichlet_noise(self) -> None:
        moves = list(self.root.children.keys())
        n = len(moves)
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_eps
        for i, m in enumerate(moves):
            child = self.root.children[m]
            child.P = (1 - eps) * child.P + eps * float(noise[i])

    def select_leaf(self) -> tuple[Optional[list[Node]], Optional[float]]:
        """
        ルートから葉まで降りる。
        Returns:
          (path, value)
          - 終局に到達した場合: (path, value) で value は確定値
          - 展開が必要な葉に到達: (path, None) で pending にセット
        """
        path = [self.root]
        current = self.root

        while not current.is_leaf():
            current = current.best_child()
            path.append(current)

        # 終局チェック
        last_move = current.move
        if last_move is not None:
            r, c = divmod(last_move, BOARD_SIZE)
            last_stone = 3 - current.to_play
            if check_winner(current.board, r, c, last_stone):
                return path, -1.0
            if not get_legal_moves(current.board):
                return path, 0.0

        # 展開が必要
        self.pending_path = path
        self.pending_leaf = current
        return path, None

    def backprop(self, path: list[Node], value: float) -> None:
        v = value
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v


class ParallelMCTS:
    """複数ゲームのMCTSをバッチ推論で並列処理する"""

    def __init__(self, net: GomokuNet, device: torch.device,
                 n_sim: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25,
                 add_noise: bool = True):
        self.net = net
        self.device = device
        self.n_sim = n_sim
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise

    def _batch_infer(self, boards: list[np.ndarray], to_plays: list[int]):
        """まとめてNN推論。Returns: (policies, values)"""
        if not boards:
            return [], []
        x = GomokuNet.batch_to_tensor(boards, to_plays, self.device)
        self.net.eval()
        with torch.no_grad():
            policies, values = self.net(x)
        policies = policies.cpu().numpy()       # (N, 225)
        values = values.cpu().numpy().flatten()  # (N,)
        return policies, values

    def _expand_node(self, node: Node, policy: np.ndarray) -> None:
        """1ノードを展開（policyは推論済みの生policy）"""
        legal = get_legal_moves(node.board)
        if not legal:
            node.is_expanded = True
            return

        masked = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        masked[legal] = policy[legal]
        s = masked.sum()
        if s > 0:
            masked /= s
        else:
            masked[legal] = 1.0 / len(legal)

        next_to_play = 3 - node.to_play
        for m in legal:
            r, c = divmod(m, BOARD_SIZE)
            new_board = node.board.copy()
            new_board[r][c] = node.to_play
            node.children[m] = Node(
                board=new_board,
                to_play=next_to_play,
                parent=node,
                move=m,
                prior=float(masked[m]),
            )
        node.is_expanded = True

    def get_action_probs_batch(
        self,
        boards: list[np.ndarray],
        to_plays: list[int],
        temperature: float = 1.0,
    ) -> list[np.ndarray]:
        """
        複数ゲームの盤面を受け取り、各ゲームの着手確率リストを返す。
        全ゲームのMCTSをバッチ推論で並列処理する。
        """
        n_games = len(boards)
        games = [
            GameMCTS(boards[i], to_plays[i],
                     self.dirichlet_alpha, self.dirichlet_eps, self.add_noise)
            for i in range(n_games)
        ]

        # ── ルート展開（全ゲーム一括） ──────────────────
        root_boards = [g.root.board for g in games]
        root_to_plays = [g.root.to_play for g in games]
        policies, values = self._batch_infer(root_boards, root_to_plays)
        for i, g in enumerate(games):
            self._expand_node(g.root, policies[i])
            g.root_expanded = True
            if g.add_noise:
                g.add_dirichlet_noise()

        # ── シミュレーションループ ──────────────────────
        for _ in range(self.n_sim):
            # 各ゲームで葉まで降りる
            infer_boards = []
            infer_to_plays = []
            infer_game_idx = []  # どのゲームの推論待ちか

            for gi, g in enumerate(games):
                path, value = g.select_leaf()
                if value is not None:
                    # 終局に到達 → すぐbackprop
                    g.backprop(path, value)
                    g.pending_path = None
                    g.pending_leaf = None
                else:
                    # 推論待ち
                    leaf = g.pending_leaf
                    infer_boards.append(leaf.board)
                    infer_to_plays.append(leaf.to_play)
                    infer_game_idx.append(gi)

            # まとめて推論
            if infer_boards:
                policies, values = self._batch_infer(infer_boards, infer_to_plays)
                for k, gi in enumerate(infer_game_idx):
                    g = games[gi]
                    leaf = g.pending_leaf
                    self._expand_node(leaf, policies[k])
                    g.backprop(g.pending_path, float(values[k]))
                    g.pending_path = None
                    g.pending_leaf = None

        # ── 各ゲームの着手確率を計算 ────────────────────
        results = []
        for g in games:
            counts = np.zeros(BOARD_SIZE * BOARD_SIZE)
            for move, child in g.root.children.items():
                counts[move] = child.N

            if temperature == 0:
                probs = np.zeros_like(counts)
                probs[np.argmax(counts)] = 1.0
            else:
                counts = counts ** (1.0 / temperature)
                s = counts.sum()
                probs = counts / s if s > 0 else np.ones_like(counts) / len(counts)
            results.append(probs)

        return results


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")
    net = GomokuNet()

    pmcts = ParallelMCTS(net, device, n_sim=20, add_noise=True)

    # 3ゲーム分の盤面を用意
    boards = []
    to_plays = []
    for i in range(3):
        b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        b[7][7] = 1  # 黒が天元
        boards.append(b)
        to_plays.append(2)  # 白の番

    print("3ゲーム並列でMCTS実行...")
    results = pmcts.get_action_probs_batch(boards, to_plays, temperature=1.0)
    print(f"結果: {len(results)} ゲーム分")
    for i, probs in enumerate(results):
        top = np.argmax(probs)
        r, c = divmod(top, BOARD_SIZE)
        print(f"  ゲーム{i}: 最有力手=({c},{r}) prob={probs[top]:.4f} sum={probs.sum():.4f}")