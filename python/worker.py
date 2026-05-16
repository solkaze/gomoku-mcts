"""
自己対局ワーカープロセスのエントリポイント

メインプロセスから呼ばれ、n_games局を自己対局して
結果をresult_queueに送る。
"""

import numpy as np
from typing import NamedTuple
from inference_server import InferenceClient
from worker_mcts import WorkerMCTS
from mcts import check_winner, get_legal_moves
from network import BOARD_SIZE


class GameSample(NamedTuple):
    board: np.ndarray
    to_play: int
    mcts_probs: np.ndarray
    value: float


class _GameState:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play = 1
        self.history = []
        self.finished = False
        self.winner = 0


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
):
    """
    ワーカープロセスのメインループ。
    n_games局を自己対局し、結果をsample_queueに送る。
    """
    client = InferenceClient(worker_id, req_queue, result_queue)
    mcts = WorkerMCTS(
        client, n_simulations,
        dirichlet_alpha, dirichlet_eps, add_noise=True,
    )

    # n_games を parallel_inner ずつ処理
    completed = 0
    while completed < n_games:
        batch = min(parallel_inner, n_games - completed)
        samples = _play_games(mcts, batch, temperature_threshold, max_moves)
        sample_queue.put((worker_id, samples))
        completed += batch

    # 完了シグナル
    sample_queue.put((worker_id, None))


def _play_games(mcts, n_games, temperature_threshold, max_moves):
    games = [_GameState() for _ in range(n_games)]

    for move_idx in range(max_moves):
        active = [g for g in games if not g.finished]
        if not active:
            break

        temp = 1.0 if move_idx < temperature_threshold else 0.0
        boards = [g.board for g in active]
        to_plays = [g.to_play for g in active]
        probs_list = mcts.get_action_probs_batch(boards, to_plays, temperature=temp)

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

    samples = []
    for g in games:
        for hist_board, hist_to_play, hist_probs in g.history:
            if g.winner == 0:
                value = 0.0
            elif g.winner == hist_to_play:
                value = 1.0
            else:
                value = -1.0
            samples.append(GameSample(hist_board, hist_to_play, hist_probs, value))
    return samples
