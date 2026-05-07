"""
MCTS (AlphaZero方式) - 修正版

設計:
  ・盤面は常に「黒=1, 白=2」のまま扱う（視点変換しない）
  ・各ノードは「次に打つ側 to_play」を保持する
  ・NN呼び出し時にboard_to_tensor(board, to_play)で視点変換
  ・backpropは標準実装（reverseしながら符号反転、加算前に反転）
"""

import math
from typing import Optional

import numpy as np
import torch
from network import BOARD_SIZE, GomokuNet

C_PUCT = 1.5
N_SIMULATIONS = 200


# ── 盤面ユーティリティ ────────────────────────────────────────

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def check_winner(board: np.ndarray, row: int, col: int, stone: int) -> bool:
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
    __slots__ = (
        "board",
        "to_play",
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
        board: np.ndarray,  # shape (15,15)  0=空 1=黒 2=白
        to_play: int,  # このノードで打つ側: 1=黒 or 2=白
        parent: Optional["Node"] = None,
        move: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.to_play = to_play
        self.parent = parent
        self.move = move
        self.children: dict[int, "Node"] = {}

        self.P = prior
        self.N = 0
        self.W = 0.0
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def ucb(self, parent_n: int) -> float:
        return self.Q + C_PUCT * self.P * math.sqrt(parent_n) / (1 + self.N)

    def best_child(self) -> "Node":
        parent_n = self.N
        return max(self.children.values(), key=lambda c: c.ucb(parent_n))

    def is_leaf(self) -> bool:
        return not self.is_expanded


# ── MCTS ────────────────────────────────────────────────────


class MCTS:
    def __init__(
        self,
        net: GomokuNet,
        device: torch.device,
        n_sim: int = N_SIMULATIONS,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        add_noise: bool = True,
    ):
        self.net = net
        self.device = device
        self.n_sim = n_sim
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise

    def get_action_probs(
        self,
        board: np.ndarray,
        to_play: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        盤面を受け取り、225マス分の着手確率を返す。
        boardは黒=1,白=2 のままでOK。to_playが次に打つ側。
        """
        root = Node(board.copy(), to_play)
        self._expand(root)

        # ルートノードの子のpriorにDirichletノイズを加える（AlphaZero標準）
        # これによりMCTSが多様な手を探索し、未知の局面で防御手などを発見しやすくなる
        if self.add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_sim):
            self._simulate(root)

        counts = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for move, child in root.children.items():
            counts[move] = child.N

        if temperature == 0:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            s = counts.sum()
            if s > 0:
                probs = counts / s
            else:
                probs = np.ones_like(counts) / len(counts)

        return probs

    def best_move(self, board: np.ndarray, to_play: int) -> int:
        probs = self.get_action_probs(board, to_play, temperature=0)
        return int(np.argmax(probs))

    def _add_dirichlet_noise(self, root: "Node") -> None:
        """ルートの子ノードのpriorにDirichletノイズを混合する"""
        moves = list(root.children.keys())
        n = len(moves)
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_eps
        for i, m in enumerate(moves):
            child = root.children[m]
            child.P = (1 - eps) * child.P + eps * float(noise[i])

    # ── 内部処理 ────────────────────────────────────────────

    def _simulate(self, root: Node) -> None:
        """Selection → Expansion or 終局 → Backpropagation"""
        path = [root]  # ★ ルートも含める

        # Selection: 葉まで降りる
        current = root
        while not current.is_leaf():
            current = current.best_child()
            path.append(current)

        # 終局チェック: 直前の手で勝ち負けが決まっているか
        # current.to_play が「これから打つ側」なので、直前に打ったのは相手
        last_move = current.move
        if last_move is not None:
            r, c = divmod(last_move, BOARD_SIZE)
            last_stone = 3 - current.to_play
            if check_winner(current.board, r, c, last_stone):
                # 直前手で勝敗確定 → 「これから打つ側(current.to_play)」から見て -1
                value = -1.0
                self._backprop(path, value)
                return

        if not get_legal_moves(current.board):
            # 引き分け
            self._backprop(path, 0.0)
            return

        # Expansion + NN評価
        value = self._expand(current)
        self._backprop(path, value)

    def _expand(self, node: Node) -> float:
        """
        ノードを展開してNN評価。
        Returns: node.to_play の視点から見た価値 [-1, 1]
        """
        x = GomokuNet.board_to_tensor(node.board, node.to_play, self.device)

        self.net.eval()
        with torch.no_grad():
            policy, value = self.net(x)

        policy = policy[0].cpu().numpy()
        value = value[0].item()

        legal = get_legal_moves(node.board)
        if not legal:
            node.is_expanded = True
            return value

        # 合法手のみで再正規化
        mask = np.zeros(BOARD_SIZE * BOARD_SIZE)
        mask[legal] = 1.0
        policy = policy * mask
        s = policy.sum()
        if s > 0:
            policy /= s
        else:
            policy[legal] = 1.0 / len(legal)

        next_to_play = 3 - node.to_play
        for m in legal:
            r, c = divmod(m, BOARD_SIZE)
            new_board = node.board.copy()
            new_board[r][c] = node.to_play  # ★ 黒なら黒石、白なら白石（視点変換しない）
            node.children[m] = Node(
                board=new_board,
                to_play=next_to_play,
                parent=node,
                move=m,
                prior=float(policy[m]),
            )

        node.is_expanded = True
        return value

    def _backprop(self, path: list[Node], value: float) -> None:
        """
        AlphaZero標準のbackprop:
          末端ノードは「自分の視点」で価値を保持
          親に上がるたびに視点が反転（手番交代）

        valueは「path末端のto_playから見た価値」として渡される。
        各ノード node に対して: node.W += (nodeのto_play視点での価値)
        nodeを上がるたびに視点反転。
        """
        # 末端から順に更新（reversedでルートに向かう）
        # 末端ノード自身は「次にそのノードのto_playが打つ」位置
        # value は「末端のto_play視点」なので、末端で W += value
        # 1つ親に上がると視点が反転する
        v = value
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")
    net = GomokuNet()
    mcts = MCTS(net, device, n_sim=50)

    # 黒が天元に打った状態 → 白の番
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1  # 黒

    print("黒が(7,7)に打った後、白(後手)の最善手を探索...")
    move = mcts.best_move(board, to_play=2)
    r, c = divmod(move, BOARD_SIZE)
    print(f"白の最善手: ({c}, {r})")
