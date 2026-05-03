"""
リプレイバッファ + データ拡張 + ディスク永続化
五目並べは8つの対称性（回転4 × 反転2）を持つので、
1つのサンプルから8倍に水増しできる。
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
from network import BOARD_SIZE, GomokuNet
from self_play import GameSample
from torch.utils.data import Dataset

# ── データ拡張 ──────────────────────────────────────────────────


def _augment(board: np.ndarray, probs: np.ndarray):
    """8通りの対称変換を適用してサンプルを生成"""
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
    """直近N局面分を保持。古いものから捨てる。"""

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
                        is_first_player=s.is_first_player,
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
        """
        バッファを単一ファイルに保存（圧縮npz）。
        各フィールドを縦に積んで配列として保存することで読み書きが速い。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.buffer)
        if n == 0:
            return

        # 配列に詰め直す
        boards = np.empty((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        is_first = np.empty(n, dtype=np.bool_)
        probs = np.empty((n, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
        values = np.empty(n, dtype=np.float32)

        for i, s in enumerate(self.buffer):
            boards[i] = s.board
            is_first[i] = s.is_first_player
            probs[i] = s.mcts_probs
            values[i] = s.value

        # 一時ファイルに書いてからリネーム（書き込み中にプロセス停止しても破損しない）
        # np.savez_compressedは拡張子.npzを自動付与するので tmp名はnpz拡張子なしで指定
        tmp_base = path.parent / (path.name + ".tmp")
        np.savez_compressed(
            tmp_base,
            boards=boards,
            is_first=is_first,
            probs=probs,
            values=values,
            capacity=np.int64(self.capacity),
        )
        # 実際に書かれたファイルは .tmp.npz になる
        actual_tmp = tmp_base.parent / (tmp_base.name + ".npz")
        actual_tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """保存したバッファを読み込む"""
        path = Path(path)
        if not path.exists():
            return cls(capacity=200_000)

        data = np.load(path)
        capacity = int(data["capacity"])
        buf = cls(capacity=capacity)

        boards = data["boards"]
        is_first = data["is_first"]
        probs = data["probs"]
        values = data["values"]

        for i in range(len(boards)):
            buf.buffer.append(
                GameSample(
                    board=boards[i],
                    is_first_player=bool(is_first[i]),
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
        x = GomokuNet.board_to_tensor(s.board, s.is_first_player).squeeze(0)
        target_probs = torch.from_numpy(s.mcts_probs)
        target_value = torch.tensor([s.value], dtype=torch.float32)
        return x, target_probs, target_value


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # 拡張テスト
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1
    board[7][8] = 2
    probs = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    probs[7 * BOARD_SIZE + 9] = 1.0
    sample = GameSample(board=board, is_first_player=True, mcts_probs=probs, value=1.0)

    buf = ReplayBuffer(capacity=100)
    buf.add_samples([sample], augment=True)
    print(f"1サンプル → {len(buf)} サンプルに拡張")

    # 永続化テスト
    print("\n--- 永続化テスト ---")
    big_buf = ReplayBuffer(capacity=10_000)
    for _ in range(1000):
        big_buf.add_samples([sample], augment=True)
    print(f"バッファサイズ: {len(big_buf)}")

    t0 = time.time()
    big_buf.save("/tmp/buf_test.npz")
    save_time = time.time() - t0

    file_size = Path("/tmp/buf_test.npz").stat().st_size
    print(f"保存: {save_time:.2f}秒, ファイルサイズ: {file_size / 1024:.1f} KB")

    t0 = time.time()
    loaded = ReplayBuffer.load("/tmp/buf_test.npz")
    load_time = time.time() - t0
    print(f"読み込み: {load_time:.2f}秒, サイズ: {len(loaded)}")

    # 内容が一致するか
    assert len(loaded) == len(big_buf)
    assert loaded.capacity == big_buf.capacity
    s_orig = list(big_buf.buffer)[0]
    s_load = list(loaded.buffer)[0]
    assert np.array_equal(s_orig.board, s_load.board)
    assert s_orig.value == s_load.value
    print("内容一致: OK")
