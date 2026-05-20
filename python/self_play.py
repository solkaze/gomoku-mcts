"""
自己対局モジュール - マルチプロセス版

構造:
  推論サーバープロセス × 1
  ワーカープロセス × N (CPUコア活用)
"""

import logging
import time
import torch
import torch.multiprocessing as mp
from typing import NamedTuple

import numpy as np
from network import GomokuNet, BOARD_SIZE
from inference_server import inference_server_loop, CMD_STOP
from worker import worker_loop
from config import Config

log = logging.getLogger("train")


class GameSample(NamedTuple):
    board: np.ndarray
    to_play: int
    mcts_probs: np.ndarray
    value: float


def play_games_parallel(
    net: GomokuNet,
    cfg: Config,
    device: torch.device,
    n_games: int,
    progress_every: int = 10,
    on_games_done=None,
) -> list[GameSample]:
    """
    n_games 局をマルチプロセスで並列自己対局する。

    progress_every: 何局完了するごとに進捗コールバックを呼ぶか
    on_games_done:  (completed: int, elapsed: float) -> None
                    標準出力への進捗表示はここで行う
    """
    tmp_model_path = "/tmp/_worker_model.bin"
    net.save_binary(tmp_model_path)

    n_workers = cfg.num_workers
    games_per_worker = (n_games + n_workers - 1) // n_workers

    ctx = mp.get_context("spawn")
    req_queue    = ctx.Queue()
    result_queues = {wid: ctx.Queue() for wid in range(n_workers)}
    sample_queue  = ctx.Queue()

    server_proc = ctx.Process(
        target=inference_server_loop,
        args=(req_queue, result_queues, tmp_model_path, str(device)),
        kwargs={
            "batch_wait_ms":  cfg.infer_batch_wait_ms,
            "max_batch_size": cfg.infer_max_batch,
            "log_path":       cfg.log_path,
        },
    )
    server_proc.start()

    worker_procs = []
    for wid in range(n_workers):
        my_games = (
            games_per_worker if wid < n_workers - 1
            else n_games - games_per_worker * (n_workers - 1)
        )
        if my_games <= 0:
            continue
        p = ctx.Process(
            target=worker_loop,
            args=(wid, req_queue, result_queues[wid], sample_queue,
                  my_games, cfg.n_simulations, cfg.parallel_inner,
                  cfg.dirichlet_alpha, cfg.dirichlet_eps,
                  cfg.temperature_threshold, cfg.max_moves,
                  cfg.log_path),
        )
        p.start()
        worker_procs.append((wid, p))

    # ── 結果を集める ────────────────────────────────────────
    all_samples   = []
    workers_done  = 0
    games_done    = 0
    n_active      = len(worker_procs)
    t_start       = time.time()
    last_reported = 0

    while workers_done < n_active:
        wid, samples = sample_queue.get()
        if samples is None:
            workers_done += 1
        else:
            all_samples.extend(samples)
            # worker.py は n_games 局分まとめて1回送るので
            # サンプル数から対局数を逆算はせず、ワーカー完了時にカウント
            # →ワーカー1台が担当分完了したタイミングで進捗を出す
            games_done += games_per_worker
            games_done  = min(games_done, n_games)

            if on_games_done and (games_done - last_reported) >= progress_every:
                on_games_done(games_done, time.time() - t_start)
                last_reported = games_done

    for _, p in worker_procs:
        p.join()
    req_queue.put((CMD_STOP,))
    server_proc.join(timeout=10)
    if server_proc.is_alive():
        server_proc.terminate()

    return all_samples


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    from logger import setup_main_logger
    setup_main_logger()

    cfg = Config(
        n_simulations=10, max_moves=20, temperature_threshold=10,
        num_workers=2, parallel_inner=2, device="cpu",
    )
    device = torch.device(cfg.device)
    net = GomokuNet().to(device)

    import time
    t0 = time.time()
    samples = play_games_parallel(net, cfg, device, n_games=4)
    print(f"サンプル数: {len(samples)}  ({time.time()-t0:.1f}秒)")
    