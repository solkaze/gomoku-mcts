"""
MCTS (Monte Carlo Tree Search) - AlphaZero方式

選択式 (PUCT):
    UCB(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Q: 平均価値（勝率）
    P: NNのPolicyによる事前確率
    N: 親ノードの訪問回数
    c_puct: 探索度合いの定数
"""

import math
from typing import Optional

import numpy as np
import torch
from network import BOARD_SIZE, GomokuNet

C_PUCT = 1.5  # 探索の強さ（大きいほど未探索を優先）
N_SIMULATIONS = 200  # 1手あたりのシミュレーション回数


# ── 盤面ユーティリティ ────────────────────────────────────────

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def check_winner(board: np.ndarray, row: int, col: int, stone: int) -> bool:
    """指定マスを起点に5連があるか確認"""
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == stone:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 5:
            return True
    return False


def get_legal_moves(board: np.ndarray) -> list[int]:
    """空きマスのインデックス（0〜224）を返す"""
    return [
        r * BOARD_SIZE + c
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
        if board[r][c] == 0
    ]


# ── ノード ───────────────────────────────────────────────────


class Node:
    """MCTSの1ノード = 1局面"""

    __slots__ = (
        "board",
        "stone",
        "parent",
        "move",
        "children",
        "P",
        "N",
        "W",
        "is_expanded",
    )

    def __init__(
        self,
        board: np.ndarray,  # shape (15,15)  0=空 1=先手 2=後手
        stone: int,  # このノードで手を打つ側: 1 or 2
        parent: Optional["Node"] = None,
        move: Optional[int] = None,  # 親から来た手（フラットインデックス）
        prior: float = 0.0,  # NNのPolicy値
    ):
        self.board = board
        self.stone = stone
        self.parent = parent
        self.move = move
        self.children: dict[int, "Node"] = {}  # move → Node

        self.P = prior  # 事前確率（NNから）
        self.N = 0  # 訪問回数
        self.W = 0.0  # 累計価値
        self.is_expanded = False

    @property
    def Q(self) -> float:
        """平均価値"""
        return self.W / self.N if self.N > 0 else 0.0

    def ucb(self, parent_n: int) -> float:
        """PUCT スコア"""
        return self.Q + C_PUCT * self.P * math.sqrt(parent_n) / (1 + self.N)

    def best_child(self) -> "Node":
        """UCBが最大の子を返す"""
        parent_n = self.N
        return max(self.children.values(), key=lambda c: c.ucb(parent_n))

    def is_leaf(self) -> bool:
        return not self.is_expanded


# ── MCTS ────────────────────────────────────────────────────


class MCTS:
    def __init__(
        self, net: GomokuNet, device: torch.device, n_sim: int = N_SIMULATIONS
    ):
        self.net = net
        self.device = device
        self.n_sim = n_sim

    # ── メインAPI ───────────────────────────────────────────

    def get_action_probs(
        self,
        board: np.ndarray,
        stone: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        盤面を受け取り、225マス分の着手確率を返す。
        temperature=1.0: 訪問回数に比例（学習時）
        temperature→0  : 最多訪問手に集中（対局時）
        """
        root = Node(board.copy(), stone)
        self._expand(root)

        for _ in range(self.n_sim):
            self._simulate(root)

        counts = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for move, child in root.children.items():
            counts[move] = child.N

        if temperature == 0:
            # 最多訪問手だけ確率1
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()

        return probs

    def best_move(self, board: np.ndarray, stone: int) -> int:
        """最善手のフラットインデックスを返す（対局用）"""
        probs = self.get_action_probs(board, stone, temperature=0)
        return int(np.argmax(probs))

    # ── 内部処理 ────────────────────────────────────────────

    def _simulate(self, node: Node) -> None:
        """Selection → Expansion → Backpropagation"""
        path = []

        # Selection: 葉ノードまで降りる
        current = node
        while not current.is_leaf():
            current = current.best_child()
            path.append(current)

        # 終局チェック（直前の手で勝敗がついているか）
        last_move = current.move
        if last_move is not None:
            r, c = divmod(last_move, BOARD_SIZE)
            # current.stone は「これから打つ側」なので相手石を確認
            last_stone = 3 - current.stone
            if check_winner(current.board, r, c, last_stone):
                # 負け局面: 打った側から見て -1
                value = -1.0
                self._backprop(path, value)
                return

        if np.all(current.board != 0):
            # 引き分け
            self._backprop(path, 0.0)
            return

        # Expansion + NNによる評価
        value = self._expand(current)
        self._backprop(path, value)

    def _expand(self, node: Node) -> float:
        """
        ノードを展開してNNで評価。
        Valueを返す（現在の手番から見た勝率）。
        """
        is_first = node.stone == 1
        x = GomokuNet.board_to_tensor(node.board, is_first, self.device)

        self.net.eval()
        with torch.no_grad():
            policy, value = self.net(x)

        policy = policy[0].cpu().numpy()  # (225,)
        value = value[0].item()  # スカラー

        # 合法手のみに絞ってPolicy再正規化
        legal = get_legal_moves(node.board)
        if not legal:
            node.is_expanded = True
            return value

        mask = np.zeros(BOARD_SIZE * BOARD_SIZE)
        mask[legal] = 1.0
        policy = policy * mask
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            # 全て0になった場合は一様分布
            policy[legal] = 1.0 / len(legal)

        # 子ノードを生成
        next_stone = 3 - node.stone
        for move in legal:
            r, c = divmod(move, BOARD_SIZE)
            new_board = node.board.copy()
            new_board[r][c] = node.stone
            node.children[move] = Node(
                board=new_board,
                stone=next_stone,
                parent=node,
                move=move,
                prior=float(policy[move]),
            )

        node.is_expanded = True
        return value

    def _backprop(self, path: list[Node], value: float) -> None:
        """
        価値を逆伝播する。
        手番が交互なので1つ上に戻るたびに符号を反転。
        """
        for node in reversed(path):
            node.N += 1
            node.W += value
            value = -value  # 手番交代で視点が反転


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")
    net = GomokuNet()
    mcts = MCTS(net, device, n_sim=50)

    # 空盤面から1手選ばせる
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1  # 先手が天元に打った後
    stone = 2  # 後手の番

    print("盤面（7,7に先手の石）から後手の最善手を探索中...")
    move = mcts.best_move(board, stone)
    r, c = divmod(move, BOARD_SIZE)
    print(f"最善手: ({c}, {r})  [フラットindex={move}]")

    # 学習用の確率分布も確認
    probs = mcts.get_action_probs(board, stone, temperature=1.0)
    top3 = np.argsort(probs)[-3:][::-1]
    print("Top-3 着手確率:")
    for idx in top3:
        r2, c2 = divmod(idx, BOARD_SIZE)
        print(f"  ({c2}, {r2}): {probs[idx]:.4f}")
