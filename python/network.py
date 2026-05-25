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

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
from pathlib import Path

log = logging.getLogger("train")

BOARD_SIZE    = 15
IN_CHANNELS   = 3
NUM_FILTERS   = 64
NUM_RES_BLOCKS = 5


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class GomokuNet(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        filters:     int = NUM_FILTERS,
        res_blocks:  int = NUM_RES_BLOCKS,
        board_size:  int = BOARD_SIZE,
    ):
        super().__init__()
        self.board_size = board_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(res_blocks)])

        self.policy_conv = nn.Conv2d(filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * board_size * board_size, board_size * board_size)

        self.value_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(board_size * board_size, 64)
        self.value_fc2  = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # 学習時はlogitsを返し、train.py側でlog_softmaxを使って数値安定に計算する
        # 推論時（eval mode）は softmax 済み確率を返す
        policy = p if self.training else F.softmax(p, dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

    @staticmethod
    def board_to_tensor(board, to_play, device=None):
        opp  = 3 - to_play
        ch0  = (board == to_play).astype(np.float32)
        ch1  = (board == opp).astype(np.float32)
        ch2  = np.ones_like(ch0) if to_play == 1 else np.zeros_like(ch0)
        tensor = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @staticmethod
    def batch_to_tensor(boards, to_plays, device=None):
        n     = len(boards)
        batch = np.empty((n, 3, boards[0].shape[0], boards[0].shape[1]), dtype=np.float32)
        for i, (board, to_play) in enumerate(zip(boards, to_plays)):
            opp          = 3 - to_play
            batch[i, 0]  = (board == to_play).astype(np.float32)
            batch[i, 1]  = (board == opp).astype(np.float32)
            batch[i, 2]  = 1.0 if to_play == 1 else 0.0
        tensor = torch.from_numpy(batch)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def save_binary(self, path) -> None:
        path  = Path(path)
        state = self.state_dict()
        with open(path, "wb") as f:
            f.write(b"GNET")
            f.write(struct.pack("<I", len(state)))
            for name, tensor in state.items():
                arr        = tensor.cpu().numpy().astype(np.float32)
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", len(arr.shape)))
                for s in arr.shape:
                    f.write(struct.pack("<I", s))
                f.write(arr.tobytes())
        total_params = sum(t.numel() for t in state.values())
        # save_binary はサブプロセスからも呼ばれるためロガー名を固定しない
        logging.getLogger("train").debug(
            f"保存: {len(state)} layers, {total_params:,} params → {path}"
        )

    @classmethod
    def load_binary(cls, path, device=None):
        path  = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GNET", f"Invalid magic: {magic}"
            num_layers = struct.unpack("<I", f.read(4))[0]
            state = {}
            for _ in range(num_layers):
                name_len = struct.unpack("<I", f.read(4))[0]
                name     = f.read(name_len).decode("utf-8")
                ndim     = struct.unpack("<I", f.read(4))[0]
                shape    = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
                num_elems = 1
                for s in shape:
                    num_elems *= s
                data     = np.frombuffer(f.read(num_elems * 4), dtype=np.float32).reshape(shape)
                state[name] = torch.from_numpy(data.copy())
        filters = state["stem.0.weight"].shape[0]
        num_res = len({k.split(".")[1] for k in state if k.startswith("res_blocks.")})
        model = cls(filters=filters, res_blocks=num_res)
        model.load_state_dict(state)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model


if __name__ == "__main__":
    from logger import setup_main_logger
    setup_main_logger()

    device = torch.device("cpu")
    net    = GomokuNet().to(device)
    total  = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,}")

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7][7] = 1
    x = GomokuNet.board_to_tensor(board, to_play=1, device=device)
    net.eval()
    with torch.no_grad():
        policy, value = net(x)
    print(f"Policy shape: {policy.shape}, sum: {policy.sum().item():.6f}")
    print(f"Value: {value.item():.4f}")
    