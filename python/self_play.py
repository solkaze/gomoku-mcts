"""
自己対局モジュール - 修正版

設計:
  ・盤面は常に「黒=1, 白=2」のまま保持
  ・MCTS呼び出し時に to_play を渡すだけ
  ・GameSampleには (盤面, to_play, mcts_probs, value) を保存
  ・valueはその局面の「to_play視点」での最終勝敗
"""

from typing import NamedTuple

import numpy as np
import torch

from config import Config
from mcts import MCTS, check_winner, get_legal_moves
from network import BOARD_SIZE, GomokuNet


class GameSample(NamedTuple):
    """学習用の1局面サンプル"""

    board: np.ndarray  # (15, 15) int8: 0=空, 1=黒, 2=白（生の盤面）
    to_play: int  # この局面で打つ側: 1=黒 or 2=白
    mcts_probs: np.ndarray  # (225,) MCTSの着手確率
    value: float  # to_play視点での最終勝敗 [-1, 1]


def play_one_game(
    net: GomokuNet, cfg: Config, device: torch.device
) -> list[GameSample]:
    """1局自己対局して、学習サンプルのリストを返す"""

    mcts = MCTS(net, device, n_sim=cfg.n_simulations)

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    to_play = 1  # 黒(先手)から開始

    # (盤面, to_play, mcts_probs) を一時保存
    history: list[tuple[np.ndarray, int, np.ndarray]] = []

    winner = 0  # 0=引き分け, 1=黒勝ち, 2=白勝ち

    for move_count in range(cfg.max_moves):
        if move_count < cfg.temperature_threshold:
            temp = 1.0
        else:
            temp = 0.0

        # MCTSで着手確率を取得（盤面は生のまま渡す）
        probs = mcts.get_action_probs(board, to_play=to_play, temperature=temp)

        history.append((board.copy(), to_play, probs))

        legal = get_legal_moves(board)
        if not legal:
            break

        if temp == 0.0:
            move = int(np.argmax(probs))
        else:
            move = int(np.random.choice(len(probs), p=probs))

        r, c = divmod(move, BOARD_SIZE)
        board[r][c] = to_play  # 黒なら1、白なら2 を直接置く

        if check_winner(board, r, c, to_play):
            winner = to_play
            break

        to_play = 3 - to_play

    # 各サンプルに勝敗結果を割り当て（to_play視点でのvalue）
    samples: list[GameSample] = []
    for hist_board, hist_to_play, hist_probs in history:
        if winner == 0:
            value = 0.0
        elif winner == hist_to_play:
            value = 1.0
        else:
            value = -1.0

        samples.append(
            GameSample(
                board=hist_board,
                to_play=hist_to_play,
                mcts_probs=hist_probs.astype(np.float32),
                value=value,
            )
        )

    return samples


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
        print(f"  最初の局面: to_play={s.to_play}, value={s.value:.1f}")
        print(f"  mcts_probs sum={s.mcts_probs.sum():.4f}")
