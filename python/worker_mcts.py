"""
ワーカープロセス用MCTS（Virtual Loss対応版）

Virtual Lossの仕組み:
  通常のMCTSは「葉まで降りる → 推論 → backprop」を1本ずつ逐次処理する。
  これではGPUへのバッチが1件になり使用率が上がらない。

  Virtual Lossは1シミュレーションステップで N_VL 本のパスを同時に降りる。
    1. パスを降りるたびに通過ノードの N+=1, W-=1 を仮加算（Virtual Loss）
       → UCBスコアが下がるので次の探索は別の経路を選ぶ
    2. 全パスの葉が揃ったらまとめてGPU推論（大きいバッチ）
    3. 実際の推論結果でbackprop し、Virtual Lossを除去

  parallel_inner=32ゲーム × N_VL=8本 = 256件/バッチ が理論上限になる。
"""

import numpy as np
from inference_server import InferenceClient, CMD_INFER
from mcts import check_winner, get_legal_moves, Node, C_PUCT

# 1シミュレーションステップで1ゲームから同時に降りるパス数
# 大きいほどバッチが太くなるが、探索の多様性が若干下がる
# 4〜8 が実用的なバランス
N_VL = 8

# Virtual Lossの重み（通過ノードに加える仮の負け数）
VIRTUAL_LOSS = 1


class GameMCTS:
    """1ゲーム分のMCTS状態"""

    def __init__(self, board: np.ndarray, to_play: int,
                 dirichlet_alpha: float, dirichlet_eps: float, add_noise: bool):
        self.root = Node(board.copy(), to_play)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise

    def add_dirichlet_noise(self):
        moves = list(self.root.children.keys())
        n = len(moves)
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_eps
        for i, m in enumerate(moves):
            child = self.root.children[m]
            child.P = (1 - eps) * child.P + eps * float(noise[i])

    def select_leaf_with_vl(self):
        """
        Virtual Lossを付けながら葉まで降りる。

        Returns:
          (path, value):
            終局に到達した場合 → value は確定値（float）
            展開が必要な葉    → value は None
        通過したノードには N+=VIRTUAL_LOSS, W-=VIRTUAL_LOSS を加えておく。
        backprop時に実際の値で上書き（除去）する。
        """
        path = [self.root]
        current = self.root

        # Virtual Lossを仮付与
        current.N += VIRTUAL_LOSS
        current.W -= VIRTUAL_LOSS

        while not current.is_leaf():
            current = current.best_child()
            path.append(current)
            current.N += VIRTUAL_LOSS
            current.W -= VIRTUAL_LOSS

        # 終局チェック
        last_move = current.move
        if last_move is not None:
            r, c = divmod(last_move, 15)
            last_stone = 3 - current.to_play
            if check_winner(current.board, r, c, last_stone):
                return path, -1.0
            if not get_legal_moves(current.board):
                return path, 0.0

        return path, None

    def backprop_with_vl(self, path: list, value: float):
        """
        Virtual Lossを除去しながら実際の値でbackpropする。
        N から VIRTUAL_LOSS を引いて正味の N に戻してから +1 する。
        W から -VIRTUAL_LOSS を引いて（つまり足して）正味の W に戻してから value を足す。
        """
        v = value
        for node in reversed(path):
            # Virtual Lossを除去してから実値を加算
            node.N += 1 - VIRTUAL_LOSS
            node.W += v + VIRTUAL_LOSS
            v = -v

    def backprop_remove_vl(self, path: list):
        """
        終局パスで推論不要だった場合でも Virtual Loss の除去だけ行う
        （backprop_with_vl と同じだが value=0 扱い → 引き分け相当）。
        実際は呼ばれないが安全策として用意。
        """
        for node in reversed(path):
            node.N -= VIRTUAL_LOSS
            node.W += VIRTUAL_LOSS


class WorkerMCTS:
    """推論サーバーを使うMCTS（ワーカープロセス内で動く）"""

    def __init__(self, client: InferenceClient, n_sim: int,
                 dirichlet_alpha: float, dirichlet_eps: float, add_noise: bool):
        self.client = client
        self.n_sim = n_sim
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise

    def _expand_node(self, node: Node, policy: np.ndarray):
        legal = get_legal_moves(node.board)
        if not legal:
            node.is_expanded = True
            return
        masked = np.zeros(225, dtype=np.float32)
        masked[legal] = policy[legal]
        s = masked.sum()
        if s > 0:
            masked /= s
        else:
            masked[legal] = 1.0 / len(legal)

        next_to_play = 3 - node.to_play
        for m in legal:
            r, c = divmod(m, 15)
            new_board = node.board.copy()
            new_board[r][c] = node.to_play
            node.children[m] = Node(
                board=new_board, to_play=next_to_play,
                parent=node, move=m, prior=float(masked[m]),
            )
        node.is_expanded = True

    def _batch_infer(self, nodes: list) -> list:
        """
        複数ノードの推論リクエストをまとめてキューに送り、まとめて受け取る。
        全リクエストを先にキューに積んでからブロック待機することで、
        サーバー側が大きなバッチを形成できる。
        """
        if not nodes:
            return []

        for node in nodes:
            self.client.req_queue.put(
                (CMD_INFER, self.client.worker_id, node.board, node.to_play)
            )

        results = []
        for _ in nodes:
            policy, value = self.client.result_queue.get()
            results.append((policy, value))

        return results

    def get_action_probs_batch(self, boards, to_plays, temperature=1.0):
        """
        複数ゲームの探索をVirtual Lossつきで実行する。

        各シミュレーションステップで全ゲームから N_VL 本ずつパスを降ろし、
        まとめてGPU推論することでバッチサイズを
          n_games × N_VL（最大）
        に拡大する。
        """
        n_games = len(boards)
        games = [
            GameMCTS(boards[i], to_plays[i],
                     self.dirichlet_alpha, self.dirichlet_eps, self.add_noise)
            for i in range(n_games)
        ]

        # ── ルート展開（全ゲームまとめてバッチ推論）──────────────
        root_results = self._batch_infer([g.root for g in games])
        for g, (policy, _) in zip(games, root_results):
            self._expand_node(g.root, policy)
            if g.add_noise:
                g.add_dirichlet_noise()

        # ── シミュレーションループ ───────────────────────────────
        # n_sim 回のシミュレーションを N_VL 本ずつまとめて処理する
        steps = (self.n_sim + N_VL - 1) // N_VL  # 切り上げ

        for _ in range(steps):
            # (game, path, value_or_None) のリスト
            pending_infer = []   # GPU推論が必要なもの
            pending_terminal = []  # 終局（即backprop）

            for g in games:
                for _ in range(N_VL):
                    path, value = g.select_leaf_with_vl()
                    if value is not None:
                        # 終局確定 → Virtual Lossを込みで即backprop
                        g.backprop_with_vl(path, value)
                    else:
                        pending_infer.append((g, path))

            if not pending_infer:
                continue

            # 推論待ちの葉をまとめてバッチ推論
            leaves = [path[-1] for _, path in pending_infer]
            infer_results = self._batch_infer(leaves)

            for (g, path), (policy, value) in zip(pending_infer, infer_results):
                leaf = path[-1]
                self._expand_node(leaf, policy)
                g.backprop_with_vl(path, float(value))

        # ── 着手確率を計算 ───────────────────────────────────────
        results = []
        for g in games:
            counts = np.zeros(225)
            for move, child in g.root.children.items():
                counts[move] = child.N
            if temperature == 0:
                probs = np.zeros_like(counts)
                probs[np.argmax(counts)] = 1.0
            else:
                counts = np.maximum(counts, 0)  # 念のため負値ガード
                counts = counts ** (1.0 / temperature)
                s = counts.sum()
                probs = counts / s if s > 0 else np.ones_like(counts) / len(counts)
            results.append(probs)
        return results
