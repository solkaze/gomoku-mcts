"""
GomokuNet: Policy + Value の2ヘッド構成
入力: (batch, 3, 15, 15)
  Ch.0 自分の石
  Ch.1 相手の石
  Ch.2 手番（定数面）
出力:
  policy: (batch, 225)  softmax済み着手確率
  value:  (batch, 1)    tanh済み勝率 [-1, 1]
"""

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 15
IN_CHANNELS = 3
NUM_FILTERS = 64
NUM_RES_BLOCKS = 5


class ResBlock(nn.Module):
    """残差ブロック: Conv → BN → ReLU → Conv → BN → residual加算 → ReLU"""

    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class GomokuNet(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        filters: int = NUM_FILTERS,
        res_blocks: int = NUM_RES_BLOCKS,
        board_size: int = BOARD_SIZE,
    ):
        super().__init__()
        self.board_size = board_size

        # ── 共通ステム ──────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(res_blocks)])

        # ── Policy Head ────────────────────────────
        self.policy_conv = nn.Conv2d(filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # ── Value Head ─────────────────────────────
        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy = F.softmax(p, dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

    # ── 盤面→テンソル変換 ───────────────────────────

    @staticmethod
    def board_to_tensor(
        board: np.ndarray,  # shape (15,15)  0=空, 1=黒, 2=白
        to_play: int,  # 次に打つ側: 1=黒(先手), 2=白(後手)
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        盤面を (1, 3, 15, 15) のテンソルに変換

        Ch.0 = to_playの石（自分の石）
        Ch.1 = 相手の石
        Ch.2 = 全1なら先手番(to_play=1)、全0なら後手番(to_play=2)

        盤面そのものは黒=1,白=2 のまま受け取り、ここで視点変換する。
        """
        opp = 3 - to_play  # 1↔2
        ch0 = (board == to_play).astype(np.float32)
        ch1 = (board == opp).astype(np.float32)
        is_first = to_play == 1
        ch2 = np.ones_like(ch0) if is_first else np.zeros_like(ch0)
        tensor = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    # ── モデルの保存・読み込み（独自バイナリ） ────────

    def save_binary(self, path: str | Path) -> None:
        """
        独自バイナリ形式で重みを保存する。

        フォーマット:
          [4 bytes] マジックナンバー "GNET"
          [4 bytes] パラメータ数 N (uint32 little-endian)
          [N × 4 bytes] float32 の重み列（各テンソルをflattenして順番に）
          [レイヤー情報] 各テンソルの (name_len, name, ndim, shape..., data) は
                         上記のフラット列に含まれる形

        シンプル版: OrderedDict順にテンソル列を書き出す。
        Rust側はニューラルネット構造を知っているので名前なし・サイズのみで十分。
        """
        path = Path(path)
        state = self.state_dict()

        with open(path, "wb") as f:
            # マジックナンバー
            f.write(b"GNET")

            # レイヤー数
            num_layers = len(state)
            f.write(struct.pack("<I", num_layers))

            for name, tensor in state.items():
                arr = tensor.cpu().numpy().astype(np.float32)
                name_bytes = name.encode("utf-8")

                # 名前の長さ + 名前
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)

                # shape の次元数 + 各次元
                shape = arr.shape
                f.write(struct.pack("<I", len(shape)))
                for s in shape:
                    f.write(struct.pack("<I", s))

                # データ本体
                f.write(arr.tobytes())

        total_params = sum(t.numel() for t in state.values())
        print(f"Saved {num_layers} layers, {total_params:,} params → {path}")

    @classmethod
    def load_binary(
        cls, path: str | Path, device: torch.device | None = None
    ) -> "GomokuNet":
        """独自バイナリから重みを復元する"""
        path = Path(path)
        model = cls()

        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GNET", f"Invalid magic: {magic}"

            num_layers = struct.unpack("<I", f.read(4))[0]
            state = {}

            for _ in range(num_layers):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")

                ndim = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))

                num_elems = 1
                for s in shape:
                    num_elems *= s
                data = np.frombuffer(f.read(num_elems * 4), dtype=np.float32).reshape(
                    shape
                )
                state[name] = torch.from_numpy(data.copy())

        model.load_state_dict(state)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model


# ── 動作確認 ────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")
    net = GomokuNet().to(device)

    # パラメータ数
    total = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,}")

    # ダミー盤面で推論
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1  # 自分の石
    board[7][8] = 2  # 相手の石
    x = GomokuNet.board_to_tensor(board, to_play=1, device=device)

    net.eval()
    with torch.no_grad():
        policy, value = net(x)

    print(f"Policy shape: {policy.shape}, sum: {policy.sum().item():.6f}")
    print(f"Value: {value.item():.4f}")
    print(f"Top-3 moves: {policy[0].topk(3).indices.tolist()}")

    # 保存・読み込みのラウンドトリップテスト
    net.save_binary("/tmp/test_model.bin")
    net2 = GomokuNet.load_binary("/tmp/test_model.bin", device=device)
    with torch.no_grad():
        policy2, value2 = net2(x)
    diff = (policy - policy2).abs().max().item()
    print(f"Round-trip max diff: {diff:.2e}  ({'OK' if diff < 1e-6 else 'NG'})")
