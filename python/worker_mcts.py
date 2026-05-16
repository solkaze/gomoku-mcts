"""
ワーカープロセス用MCTS

推論をInferenceClient経由で行う以外は、parallel_mcts.pyと同じ動作。
1ワーカーが内部で複数ゲームを並列保持し、推論はサーバーに投げる。
"""

import math
import numpy as np
from typing import Optional
from inference_server import InferenceClient
from mcts import check_winner, get_legal_moves, Node, C_PUCT


class GameMCTS:
    """1ゲーム分のMCTS状態"""
    def __init__(self, board: np.ndarray, to_play: int,
                 dirichlet_alpha: float, dirichlet_eps: float, add_noise: bool):
        self.root = Node(board.copy(), to_play)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise
        self.pending_path = None
        self.pending_leaf = None

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

    def select_leaf(self):
        path = [self.root]
        current = self.root
        while not current.is_leaf():
            current = current.best_child()
            path.append(current)

        last_move = current.move
        if last_move is not None:
            r, c = divmod(last_move, 15)
            last_stone = 3 - current.to_play
            if check_winner(current.board, r, c, last_stone):
                return path, -1.0
            if not get_legal_moves(current.board):
                return path, 0.0

        self.pending_path = path
        self.pending_leaf = current
        return path, None

    def backprop(self, path, value):
        v = value
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v


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

    def get_action_probs_batch(self, boards, to_plays, temperature=1.0):
        """複数ゲームの探索をまとめて実行（ワーカー内で並列）"""
        n_games = len(boards)
        games = [
            GameMCTS(boards[i], to_plays[i],
                     self.dirichlet_alpha, self.dirichlet_eps, self.add_noise)
            for i in range(n_games)
        ]

        # ルート展開（推論サーバーに投げてまとめて受け取る）
        for g in games:
            policy, _ = self.client.infer(g.root.board, g.root.to_play)
            self._expand_node(g.root, policy)
            if g.add_noise:
                g.add_dirichlet_noise()

        # シミュレーションループ
        for _ in range(self.n_sim):
            for g in games:
                path, value = g.select_leaf()
                if value is not None:
                    g.backprop(path, value)
                else:
                    # 葉ノードを推論サーバーで評価
                    leaf = g.pending_leaf
                    policy, value = self.client.infer(leaf.board, leaf.to_play)
                    self._expand_node(leaf, policy)
                    g.backprop(g.pending_path, value)
                    g.pending_path = None
                    g.pending_leaf = None

        # 着手確率を計算
        results = []
        for g in games:
            counts = np.zeros(225)
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
