"""
リプレイバッファ + データ拡張
五目並べは8つの対称性（回転4 × 反転2）を持つので、
1つのサンプルから8倍に水増しできる。
"""

from collections import deque

import numpy as np
import torch
from network import BOARD_SIZE, GomokuNet
from self_play import GameSample
from torch.utils.data import Dataset

# ── データ拡張 ──────────────────────────────────────────────────


def _augment(board: np.ndarray, probs: np.ndarray):
    """
    8通りの対称変換を適用してサンプルを生成。
    Returns: list of (board, probs)
    """
    augmented = []
    probs_2d = probs.reshape(BOARD_SIZE, BOARD_SIZE)

    for k in range(4):  # 90度ずつ回転
        rot_board = np.rot90(board, k).copy()
        rot_probs = np.rot90(probs_2d, k).copy()
        augmented.append((rot_board, rot_probs.flatten()))

        # 左右反転バージョン
        flip_board = np.fliplr(rot_board).copy()
        flip_probs = np.fliplr(rot_probs).copy()
        augmented.append((flip_board, flip_probs.flatten()))

    return augmented


# ── リプレイバッファ ────────────────────────────────────────────


class ReplayBuffer:
    """直近N局面分を保持。古いものから捨てる。"""

    def __init__(self, capacity: int):
        self.buffer: deque[GameSample] = deque(maxlen=capacity)

    def add_samples(self, samples: list[GameSample], augment: bool = True) -> None:
        for s in samples:
            if not augment:
                self.buffer.append(s)
                continue

            # 8倍データ拡張
            for aug_board, aug_probs in _augment(s.board, s.mcts_probs):
                self.buffer.append(
                    GameSample(
                        board=aug_board.astype(np.int8),
                        is_first_player=s.is_first_player,
                        mcts_probs=aug_probs.astype(np.float32),
                        value=s.value,
                    )
                )

    def __len__(self) -> int:
        return len(self.buffer)

    def to_dataset(self) -> "BufferDataset":
        return BufferDataset(list(self.buffer))


# ── PyTorch Dataset ─────────────────────────────────────────────


class BufferDataset(Dataset):
    """ReplayBufferの内容をPyTorchで使えるDatasetに変換"""

    def __init__(self, samples: list[GameSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # 入力テンソル化（network側のヘルパーを使用）
        x = GomokuNet.board_to_tensor(s.board, s.is_first_player).squeeze(0)
        target_probs = torch.from_numpy(s.mcts_probs)
        target_value = torch.tensor([s.value], dtype=torch.float32)
        return x, target_probs, target_value


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    # ダミーサンプルで拡張テスト
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1
    board[7][8] = 2
    probs = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    probs[7 * BOARD_SIZE + 9] = 1.0  # (7,9) に確率1

    sample = GameSample(board=board, is_first_player=True, mcts_probs=probs, value=1.0)

    buf = ReplayBuffer(capacity=100)
    buf.add_samples([sample], augment=True)
    print(f"1サンプル → {len(buf)} サンプルに拡張")

    # 8倍されているか確認
    assert len(buf) == 8

    # 各拡張版で確率の合計は1のままか
    for s in buf.buffer:
        assert abs(s.mcts_probs.sum() - 1.0) < 1e-5

    print("拡張後も確率合計=1.0 を維持: OK")

    # Dataset化
    ds = buf.to_dataset()
    x, p, v = ds[0]
    print(f"Dataset[0]: x.shape={x.shape}, p.shape={p.shape}, v={v.item()}")
