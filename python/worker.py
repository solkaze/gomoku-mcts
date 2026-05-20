"""
自己対局ワーカープロセスのエントリポイント（スロット制・logging対応版）

常に parallel_inner 本のスロットを維持し、終局したスロットを即補充することで
序盤〜終盤を通じてフルバッチを維持する。
"""

import numpy as np
from typing import NamedTuple
from inference_server import InferenceClient
from worker_mcts import WorkerMCTS
from mcts import check_winner, get_legal_moves
from network import BOARD_SIZE
from logger import setup_worker_logger


class GameSample(NamedTuple):
    board: np.ndarray
    to_play: int
    mcts_probs: np.ndarray
    value: float


class _GameState:
    def __init__(self):
        self.board    = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play  = 1
        self.history  = []
        self.finished = False
        self.winner   = 0
        self.move_idx = 0


def worker_loop(
    worker_id: int,
    req_queue,
    result_queue,
    sample_queue,
    n_games: int,
    n_simulations: int,
    parallel_inner: int,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temperature_threshold: int,
    max_moves: int,
    log_path: str = "/workspace/logs/train.log",
):
    log = setup_worker_logger(f"worker.{worker_id}", log_path)
    log.debug(f"起動: n_games={n_games} parallel_inner={parallel_inner}")

    client = InferenceClient(worker_id, req_queue, result_queue)
    mcts   = WorkerMCTS(
        client, n_simulations,
        dirichlet_alpha, dirichlet_eps, add_noise=True,
    )

    samples = _play_games_slots(
        mcts, n_games, parallel_inner, temperature_threshold, max_moves, log
    )
    log.debug(f"完了: {len(samples)} サンプル収集")

    sample_queue.put((worker_id, samples))
    sample_queue.put((worker_id, None))


def _collect_samples(g: _GameState) -> list:
    result = []
    for hist_board, hist_to_play, hist_probs in g.history:
        if g.winner == 0:
            value = 0.0
        elif g.winner == hist_to_play:
            value = 1.0
        else:
            value = -1.0
        result.append(GameSample(hist_board, hist_to_play, hist_probs, value))
    return result


def _play_games_slots(mcts, n_games, n_slots, temperature_threshold, max_moves, log):
    slots          = [_GameState() for _ in range(n_slots)]
    all_samples    = []
    games_started  = n_slots
    games_finished = 0

    while games_finished < n_games:
        active_idx   = [i for i, g in enumerate(slots) if not g.finished]
        if not active_idx:
            break

        active_games = [slots[i] for i in active_idx]
        boards       = [g.board   for g in active_games]
        to_plays     = [g.to_play for g in active_games]

        probs_list = mcts.get_action_probs_batch(boards, to_plays, temperature=1.0)

        for g, probs in zip(active_games, probs_list):
            temp = 1.0 if g.move_idx < temperature_threshold else 0.0
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
                g.winner   = g.to_play
                g.finished = True
            elif g.move_idx + 1 >= max_moves:
                g.finished = True
            else:
                g.to_play  = 3 - g.to_play
                g.move_idx += 1

        for i, g in enumerate(slots):
            if not g.finished:
                continue

            all_samples.extend(_collect_samples(g))
            games_finished += 1

            if games_finished >= n_games:
                slots[i].finished = True
            elif games_started < n_games:
                slots[i]      = _GameState()
                games_started += 1
            else:
                slots[i].finished = True

    return all_samples
