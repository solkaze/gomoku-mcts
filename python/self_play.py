"""
自己対局モジュール - 並列バッチMCTS版

複数の対局を同時進行させ、各手のMCTSをバッチ推論で高速化する。
全ゲームが「次の1手」を同時に決めるように進行する。
"""

import numpy as np
import torch
from typing import NamedTuple
from network import GomokuNet, BOARD_SIZE
from parallel_mcts import ParallelMCTS
from mcts import check_winner, get_legal_moves
from config import Config


class GameSample(NamedTuple):
    board: np.ndarray       # (15, 15) int8: 0=空, 1=黒, 2=白
    to_play: int            # この局面で打つ側: 1=黒 or 2=白
    mcts_probs: np.ndarray  # (225,) MCTSの着手確率
    value: float            # to_play視点での最終勝敗 [-1, 1]


class _GameState:
    """並列対局の1ゲーム分の進行状態"""
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play = 1  # 黒から
        self.history: list[tuple[np.ndarray, int, np.ndarray]] = []
        self.finished = False
        self.winner = 0  # 0=引き分け, 1=黒, 2=白
        self.move_count = 0


def play_games_parallel(
    net: GomokuNet,
    cfg: Config,
    device: torch.device,
    n_games: int,
) -> list[GameSample]:
    """
    n_games 局を並列に自己対局し、全局面の学習サンプルをまとめて返す。
    """
    pmcts = ParallelMCTS(
        net, device,
        n_sim=cfg.n_simulations,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        add_noise=True,
    )

    games = [_GameState() for _ in range(n_games)]

    for move_idx in range(cfg.max_moves):
        # まだ終わっていないゲームを集める
        active = [g for g in games if not g.finished]
        if not active:
            break

        # 温度設定（全ゲーム共通）
        temp = 1.0 if move_idx < cfg.temperature_threshold else 0.0

        # アクティブゲームの盤面をまとめてMCTS
        boards = [g.board for g in active]
        to_plays = [g.to_play for g in active]
        probs_list = pmcts.get_action_probs_batch(boards, to_plays, temperature=temp)

        # 各ゲームで着手
        for g, probs in zip(active, probs_list):
            g.history.append((g.board.copy(), g.to_play, probs.astype(np.float32)))

            legal = get_legal_moves(g.board)
            if not legal:
                g.finished = True
                continue

            if temp == 0.0:
                move = int(np.argmax(probs))
            else:
                move = int(np.random.choice(len(probs), p=probs))

            r, c = divmod(move, BOARD_SIZE)
            g.board[r][c] = g.to_play

            if check_winner(g.board, r, c, g.to_play):
                g.winner = g.to_play
                g.finished = True
            else:
                g.to_play = 3 - g.to_play
            g.move_count += 1

    # ── 全ゲームの履歴からサンプルを生成 ──────────────────
    samples: list[GameSample] = []
    for g in games:
        for hist_board, hist_to_play, hist_probs in g.history:
            if g.winner == 0:
                value = 0.0
            elif g.winner == hist_to_play:
                value = 1.0
            else:
                value = -1.0
            samples.append(GameSample(
                board=hist_board,
                to_play=hist_to_play,
                mcts_probs=hist_probs,
                value=value,
            ))

    return samples


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config(
        n_simulations=20, temperature_threshold=10, max_moves=30, device="cpu"
    )
    device = torch.device(cfg.device)
    net = GomokuNet().to(device)

    print("4ゲーム並列で自己対局...")
    import time
    t0 = time.time()
    samples = play_games_parallel(net, cfg, device, n_games=4)
    elapsed = time.time() - t0
    print(f"集めたサンプル数: {len(samples)}  ({elapsed:.1f}秒)")

    if samples:
        s = samples[0]
        print(f"  最初の局面: to_play={s.to_play}, value={s.value:.1f}")