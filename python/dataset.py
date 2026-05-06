"""
リプレイバッファ + データ拡張 + ディスク永続化（新設計版）

データ拡張:
  五目並べは8つの対称性（回転4 × 反転2）を持つ。
  to_playの値は変わらず、boardとmcts_probsだけ回転・反転する。
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from network import BOARD_SIZE, GomokuNet
from self_play import GameSample

# ── データ拡張 ──────────────────────────────────────────────────


def _augment(board: np.ndarray, probs: np.ndarray):
    """8通りの対称変換"""
    augmented = []
    probs_2d = probs.reshape(BOARD_SIZE, BOARD_SIZE)

    for k in range(4):
        rot_board = np.rot90(board, k).copy()
        rot_probs = np.rot90(probs_2d, k).copy()
        augmented.append((rot_board, rot_probs.flatten()))

        flip_board = np.fliplr(rot_board).copy()
        flip_probs = np.fliplr(rot_probs).copy()
        augmented.append((flip_board, flip_probs.flatten()))

    return augmented


# ── リプレイバッファ ────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[GameSample] = deque(maxlen=capacity)

    def add_samples(self, samples: list[GameSample], augment: bool = True) -> None:
        for s in samples:
            if not augment:
                self.buffer.append(s)
                continue

            for aug_board, aug_probs in _augment(s.board, s.mcts_probs):
                self.buffer.append(
                    GameSample(
                        board=aug_board.astype(np.int8),
                        to_play=s.to_play,
                        mcts_probs=aug_probs.astype(np.float32),
                        value=s.value,
                    )
                )

    def __len__(self) -> int:
        return len(self.buffer)

    def to_dataset(self) -> "BufferDataset":
        return BufferDataset(list(self.buffer))

    # ── 永続化 ─────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.buffer)
        if n == 0:
            return

        boards = np.empty((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        to_play_arr = np.empty(n, dtype=np.int8)
        probs = np.empty((n, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
        values = np.empty(n, dtype=np.float32)

        for i, s in enumerate(self.buffer):
            boards[i] = s.board
            to_play_arr[i] = s.to_play
            probs[i] = s.mcts_probs
            values[i] = s.value

        tmp_base = path.parent / (path.name + ".tmp")
        np.savez_compressed(
            tmp_base,
            boards=boards,
            to_play=to_play_arr,
            probs=probs,
            values=values,
            capacity=np.int64(self.capacity),
        )
        actual_tmp = tmp_base.parent / (tmp_base.name + ".npz")
        actual_tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        path = Path(path)
        if not path.exists():
            return cls(capacity=200_000)

        data = np.load(path)
        capacity = int(data["capacity"])
        buf = cls(capacity=capacity)

        boards = data["boards"]
        to_play_arr = data["to_play"]
        probs = data["probs"]
        values = data["values"]

        for i in range(len(boards)):
            buf.buffer.append(
                GameSample(
                    board=boards[i],
                    to_play=int(to_play_arr[i]),
                    mcts_probs=probs[i],
                    value=float(values[i]),
                )
            )

        return buf


# ── PyTorch Dataset ─────────────────────────────────────────────


class BufferDataset(Dataset):
    def __init__(self, samples: list[GameSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = GomokuNet.board_to_tensor(s.board, s.to_play).squeeze(0)
        target_probs = torch.from_numpy(s.mcts_probs)
        target_value = torch.tensor([s.value], dtype=torch.float32)
        return x, target_probs, target_value


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1
    board[7][8] = 2
    probs = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    probs[7 * BOARD_SIZE + 9] = 1.0

    sample = GameSample(board=board, to_play=1, mcts_probs=probs, value=1.0)

    buf = ReplayBuffer(capacity=100)
    buf.add_samples([sample], augment=True)
    print(f"1サンプル → {len(buf)} サンプルに拡張")
    assert len(buf) == 8

    # 永続化テスト
    buf.save("/tmp/buf_v2.npz")
    loaded = ReplayBuffer.load("/tmp/buf_v2.npz")
    assert len(loaded) == len(buf)
    assert list(loaded.buffer)[0].to_play == 1
    print("永続化OK")

    # Dataset化
    ds = buf.to_dataset()
    x, p, v = ds[0]
    print(
        f"Dataset: x.shape={x.shape}, Ch.2[0,0]={x[2, 0, 0].item()}（先手なので1.0のはず）"
    )
