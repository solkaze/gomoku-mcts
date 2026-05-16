"""
推論サーバー: GPU専属プロセスで複数ワーカーからの推論要求をバッチ処理する。

ワーカーは次のように使う:
    req_queue.put((worker_id, board, to_play))
    policy, value = result_queues[worker_id].get()

サーバーは複数の要求を貯めてからまとめてGPU推論する。
"""

import time
import numpy as np
import torch
import torch.multiprocessing as mp
from network import GomokuNet, BOARD_SIZE


# サーバープロセスへのコマンド
CMD_INFER = "infer"
CMD_RELOAD = "reload"  # モデル重み更新
CMD_STOP = "stop"


def inference_server_loop(
    req_queue: mp.Queue,
    result_queues: dict,
    model_state_dict_path: str,
    device_str: str,
    batch_wait_ms: float = 1.0,
    max_batch_size: int = 256,
):
    """
    推論サーバープロセスのメインループ。

    req_queue: 全ワーカーが投げ込む要求キュー
      要素: (cmd, worker_id, board, to_play) または (cmd, ...)
    result_queues: ワーカーIDをキーとした結果送信用キュー dict
    model_state_dict_path: 起動時に読み込むモデルファイルパス
    """
    device = torch.device(device_str)
    net = GomokuNet.load_binary(model_state_dict_path, device=device)
    net.eval()

    print(f"[InferenceServer] 起動: device={device}", flush=True)

    while True:
        # 1つ目のリクエストを待つ（ブロッキング）
        try:
            first = req_queue.get(timeout=60.0)
        except Exception:
            print("[InferenceServer] タイムアウト、終了", flush=True)
            break

        cmd = first[0]

        if cmd == CMD_STOP:
            print("[InferenceServer] 停止コマンド受信", flush=True)
            break

        if cmd == CMD_RELOAD:
            new_path = first[1]
            net = GomokuNet.load_binary(new_path, device=device)
            net.eval()
            continue

        # CMD_INFER: バッチを作る
        batch = [first]
        # 短時間だけ後続リクエストを集める
        deadline = time.time() + (batch_wait_ms / 1000.0)
        while len(batch) < max_batch_size and time.time() < deadline:
            try:
                item = req_queue.get(timeout=0.001)
            except Exception:
                break
            if item[0] == CMD_INFER:
                batch.append(item)
            elif item[0] == CMD_STOP:
                # 残った推論を完了させてから止める
                _process_batch(net, device, batch, result_queues)
                return
            else:
                batch.append(item)

        _process_batch(net, device, batch, result_queues)


def _process_batch(net, device, batch, result_queues):
    """バッチ推論を実行して結果を各ワーカーに返す"""
    worker_ids = [item[1] for item in batch]
    boards = [item[2] for item in batch]
    to_plays = [item[3] for item in batch]

    x = GomokuNet.batch_to_tensor(boards, to_plays, device=device)
    with torch.no_grad():
        policies, values = net(x)
    policies = policies.cpu().numpy()
    values = values.cpu().numpy().flatten()

    for i, wid in enumerate(worker_ids):
        result_queues[wid].put((policies[i], float(values[i])))


class InferenceClient:
    """ワーカープロセスから推論サーバーを利用するためのクライアント"""

    def __init__(self, worker_id: int, req_queue: mp.Queue, result_queue: mp.Queue):
        self.worker_id = worker_id
        self.req_queue = req_queue
        self.result_queue = result_queue

    def infer(self, board: np.ndarray, to_play: int):
        """1局面の推論を要求して結果を待つ"""
        self.req_queue.put((CMD_INFER, self.worker_id, board, to_play))
        policy, value = self.result_queue.get()
        return policy, value
