"""
自己対局モジュール - マルチプロセス版

複数のワーカープロセスで自己対局を並列実行し、
GPU推論は専属の推論サーバープロセスで一括処理する。

構造:
  推論サーバープロセス × 1
  ワーカープロセス × N (CPUコア活用)
"""

import os
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import NamedTuple

from network import GomokuNet, BOARD_SIZE
from inference_server import inference_server_loop, CMD_STOP
from worker import worker_loop
from config import Config


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
) -> list[GameSample]:
    """
    n_games 局をマルチプロセスで並列自己対局する。
    """
    # ── モデルを一時ファイルに保存（ワーカープロセス間で共有用）─────
    # netはメインプロセスのものなので、サーバープロセスは別途読み込む
    tmp_model_path = "/tmp/_worker_model.bin"
    net.save_binary(tmp_model_path)

    # ── キューを準備 ────────────────────────────────────────
    n_workers = cfg.num_workers
    games_per_worker = (n_games + n_workers - 1) // n_workers

    ctx = mp.get_context("spawn")
    req_queue = ctx.Queue()
    result_queues = {wid: ctx.Queue() for wid in range(n_workers)}
    sample_queue = ctx.Queue()

    # ── 推論サーバープロセスを起動 ──────────────────────────
    server_proc = ctx.Process(
        target=inference_server_loop,
        args=(req_queue, result_queues, tmp_model_path, str(device)),
        kwargs={"batch_wait_ms": cfg.infer_batch_wait_ms,
                "max_batch_size": cfg.infer_max_batch},
    )
    server_proc.start()

    # ── ワーカープロセスを起動 ──────────────────────────────
    worker_procs = []
    for wid in range(n_workers):
        # 各ワーカーが担当する局数（端数は最後のワーカーが調整）
        my_games = games_per_worker if wid < n_workers - 1 else (n_games - games_per_worker * (n_workers - 1))
        if my_games <= 0:
            continue
        p = ctx.Process(
            target=worker_loop,
            args=(wid, req_queue, result_queues[wid], sample_queue,
                  my_games, cfg.n_simulations, cfg.parallel_inner,
                  cfg.dirichlet_alpha, cfg.dirichlet_eps,
                  cfg.temperature_threshold, cfg.max_moves),
        )
        p.start()
        worker_procs.append((wid, p))

    # ── 結果を集める ────────────────────────────────────────
    all_samples = []
    workers_done = 0
    n_active_workers = len(worker_procs)
    while workers_done < n_active_workers:
        wid, samples = sample_queue.get()
        if samples is None:
            workers_done += 1
        else:
            all_samples.extend(samples)

    # ── 後片付け ───────────────────────────────────────────
    for _, p in worker_procs:
        p.join()
    req_queue.put((CMD_STOP,))
    server_proc.join(timeout=10)
    if server_proc.is_alive():
        server_proc.terminate()

    return all_samples


# ── 動作確認 ─────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config(
        n_simulations=10, max_moves=20, temperature_threshold=10,
        num_workers=2, parallel_inner=2, device="cpu",
    )
    device = torch.device(cfg.device)
    net = GomokuNet().to(device)

    print("4ゲームをマルチプロセスで実行...")
    import time
    t0 = time.time()
    samples = play_games_parallel(net, cfg, device, n_games=4)
    elapsed = time.time() - t0
    print(f"集めたサンプル数: {len(samples)}  ({elapsed:.1f}秒)")
    