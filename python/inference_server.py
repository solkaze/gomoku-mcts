"""
推論サーバー: GPU専属プロセスで複数ワーカーからの推論要求をバッチ処理する。
"""

import time
import numpy as np
import torch
import torch.multiprocessing as mp
from network import GomokuNet, BOARD_SIZE
from logger import setup_worker_logger

CMD_INFER  = "infer"
CMD_RELOAD = "reload"
CMD_STOP   = "stop"


def inference_server_loop(
    req_queue: mp.Queue,
    result_queues: dict,
    model_state_dict_path: str,
    device_str: str,
    batch_wait_ms: float = 5.0,
    max_batch_size: int = 1024,
    log_path: str = "/workspace/logs/train.log",
):
    log = setup_worker_logger("inference_server", log_path)

    device = torch.device(device_str)
    net = GomokuNet.load_binary(model_state_dict_path, device=device)
    net.eval()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log.info(f"起動: device={device}")

    _total_batches = 0
    _gpu_time = 0.0
    _t0 = None

    while True:
        try:
            first = req_queue.get(timeout=60.0)
        except Exception:
            log.warning("タイムアウト、終了")
            break

        cmd = first[0]

        if cmd == CMD_STOP:
            log.info("停止コマンド受信")
            break

        if cmd == CMD_RELOAD:
            net = GomokuNet.load_binary(first[1], device=device)
            net.eval()
            log.info(f"モデル再読み込み: {first[1]}")
            continue

        # バッチを貪欲に集める
        batch = [first]
        deadline = time.perf_counter() + batch_wait_ms / 1000.0
        while len(batch) < max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = req_queue.get(timeout=max(remaining, 0.0001))
            except Exception:
                break
            if item[0] == CMD_STOP:
                _do_infer(net, device, batch, result_queues)
                log.info("停止コマンド受信（バッチ処理後）")
                return
            batch.append(item)

        t0 = time.perf_counter()
        _do_infer(net, device, batch, result_queues)
        _gpu_time += time.perf_counter() - t0
        _total_batches += 1

        if _t0 is None:
            _t0 = time.perf_counter()

        if _total_batches % 500 == 0 and _t0 is not None:
            wall = time.perf_counter() - _t0
            util = _gpu_time / wall * 100 if wall > 0 else 0
            log.debug(
                f"GPU稼働率={util:.1f}%  gpu={_gpu_time:.1f}s  "
                f"wall={wall:.1f}s  batches={_total_batches}  "
                f"latest_bs={len(batch)}"
            )


def _do_infer(net, device, batch, result_queues):
    worker_ids = [item[1] for item in batch]
    boards     = [item[2] for item in batch]
    to_plays   = [item[3] for item in batch]

    x = GomokuNet.batch_to_tensor(boards, to_plays, device=device)
    with torch.no_grad():
        policies, values = net(x)

    policies = policies.cpu().numpy()
    values   = values.cpu().numpy().flatten()

    for i, wid in enumerate(worker_ids):
        result_queues[wid].put((policies[i], float(values[i])))


class InferenceClient:
    def __init__(self, worker_id: int, req_queue: mp.Queue, result_queue: mp.Queue):
        self.worker_id    = worker_id
        self.req_queue    = req_queue
        self.result_queue = result_queue

    def infer(self, board: np.ndarray, to_play: int):
        self.req_queue.put((CMD_INFER, self.worker_id, board, to_play))
        return self.result_queue.get()
