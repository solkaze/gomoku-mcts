"""
自己対局モジュール
NN + MCTS で1局を実行し、(盤面, MCTS確率分布, 勝敗) を蓄積する。
"""

from typing import NamedTuple

import numpy as np
import torch
from config import Config
from mcts import MCTS, check_winner, get_legal_moves
from network import BOARD_SIZE, GomokuNet


class GameSample(NamedTuple):
    """学習用の1局面サンプル"""

    board: np.ndarray  # (15, 15) int8: 0=空, 1=自分, 2=相手
    is_first_player: bool  # この局面で打つ側が先手か
    mcts_probs: np.ndarray  # (225,) MCTSが出した着手確率（学習のターゲット）
    value: float  # この局面で打つ側から見た最終勝敗 [-1, 1]


def play_one_game(
    net: GomokuNet, cfg: Config, device: torch.device
) -> list[GameSample]:
    """1局自己対局して、その棋譜から学習サンプルのリストを返す"""

    mcts = MCTS(net, device, n_sim=cfg.n_simulations)

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    stone = 1  # 1=先手から開始

    # 各手の (盤面, 手番, mcts確率) を一時保存
    history: list[tuple[np.ndarray, int, np.ndarray]] = []

    winner = 0  # 0=引き分け, 1=先手勝ち, 2=後手勝ち

    for move_count in range(cfg.max_moves):
        # 序盤は探索広く、終盤は確定的に
        if move_count < cfg.temperature_threshold:
            temp = 1.0
        else:
            temp = 0.0

        # MCTSで着手確率を取得（Note: stoneの視点に変換するためboard表現を切り替え）
        # MCTS内部では「自分=1, 相手=2」を期待しているので、現在手番の視点に変換
        my_view = _to_my_view(board, stone)
        probs = mcts.get_action_probs(my_view, stone=1, temperature=temp)

        # 履歴に保存（盤面は元のまま、視点情報だけ持っておく）
        history.append((board.copy(), stone, probs))

        # 着手選択
        legal = get_legal_moves(my_view)
        if not legal:
            break

        # probsから着手をサンプリング（temp=0なら最大値固定）
        if temp == 0.0:
            move = int(np.argmax(probs))
        else:
            move = int(np.random.choice(len(probs), p=probs))

        # 盤面に反映
        r, c = divmod(move, BOARD_SIZE)
        board[r][c] = stone

        # 勝敗判定
        if check_winner(board, r, c, stone):
            winner = stone
            break

        # 手番交代
        stone = 3 - stone

    # 各サンプルに勝敗結果を割り当てる
    samples: list[GameSample] = []
    for hist_board, hist_stone, hist_probs in history:
        if winner == 0:
            value = 0.0
        elif winner == hist_stone:
            value = 1.0
        else:
            value = -1.0

        # 視点を「自分=1, 相手=2」に統一
        my_view = _to_my_view(hist_board, hist_stone)
        is_first = hist_stone == 1

        samples.append(
            GameSample(
                board=my_view,
                is_first_player=is_first,
                mcts_probs=hist_probs.astype(np.float32),
                value=value,
            )
        )

    return samples


def _to_my_view(board: np.ndarray, stone: int) -> np.ndarray:
    """
    生の盤面（1=先手の石, 2=後手の石）を、
    現在手番の視点（1=自分, 2=相手）に変換する。
    """
    if stone == 1:
        return board.copy()
    # 後手視点: 1↔2を入れ替え
    view = board.copy()
    view[board == 1] = 2
    view[board == 2] = 1
    return view


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config(n_simulations=20, temperature_threshold=10, max_moves=30, device="cpu")
    device = torch.device(cfg.device)
    net = GomokuNet().to(device)

    print("自己対局を1局実行...")
    samples = play_one_game(net, cfg, device)
    print(f"集めたサンプル数: {len(samples)}")

    if samples:
        s = samples[0]
        print(f"  最初の局面: 先手={s.is_first_player}, value={s.value:.1f}")
        print(f"  mcts_probs shape={s.mcts_probs.shape}, sum={s.mcts_probs.sum():.4f}")
