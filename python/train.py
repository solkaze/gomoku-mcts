"""
学習ループのメインスクリプト

[自己対局] → [リプレイバッファ追加] → [NN学習] → [モデル保存] を繰り返す。

標準出力: イテレーション結果 + 10試合ごとの自己対局進捗のみ
ログファイル: 全デバッグ情報（inference_server統計、worker詳細など）
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from network import GomokuNet
from self_play import play_games_parallel
from dataset import ReplayBuffer
from config import Config, DEFAULT
from stats import TrainingStats
from logger import setup_main_logger


# ── 損失関数 ────────────────────────────────────────────────

def alphazero_loss(pred_policy, pred_value, target_policy, target_value):
    # pred_policy は学習時にlogitsで来る。log_softmaxで数値安定に計算する。
    log_probs   = nn.functional.log_softmax(pred_policy, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss  = nn.functional.mse_loss(pred_value, target_value)
    return policy_loss + value_loss, policy_loss, value_loss


# ── 1イテレーション ─────────────────────────────────────────

def run_iteration(net, optimizer, buffer, cfg, device, iter_idx, global_iter):
    log   = logging.getLogger("train")
    stats = {}

    # ── ① 自己対局 ───────────────────────────────────────
    t0            = time.time()
    games_done_so_far = [0]  # closure用

    def on_progress(completed: int, elapsed: float):
        games_done_so_far[0] = completed
        rate = completed / elapsed if elapsed > 0 else 0
        # 標準出力へ（INFOだがtrainロガーなので出力される）
        print(
            f"  [{completed:3d}/{cfg.games_per_iteration}試合完了]  "
            f"経過 {elapsed:.0f}s  ({rate:.1f}試合/s)",
            flush=True,
        )

    samples = play_games_parallel(
        net, cfg, device,
        n_games=cfg.games_per_iteration,
        progress_every=10,
        on_games_done=on_progress,
    )
    buffer.add_samples(samples, augment=True)
    new_samples  = len(samples)
    self_play_sec = time.time() - t0

    log.debug(
        f"自己対局完了: {cfg.games_per_iteration}局 "
        f"新サンプル={new_samples} elapsed={self_play_sec:.1f}s"
    )

    stats["self_play_time"] = self_play_sec
    stats["new_samples"]    = new_samples
    stats["buffer_size"]    = len(buffer)

    if len(buffer) < cfg.batch_size:
        log.info(f"バッファ不足({len(buffer)} < {cfg.batch_size})、学習スキップ")
        return stats

    # ── ② 学習 ───────────────────────────────────────────
    t0      = time.time()
    dataset = buffer.to_dataset()
    loader  = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    net.train()
    total_losses  = []
    policy_losses = []
    value_losses  = []

    for epoch in range(cfg.epochs_per_iteration):
        ep_total = ep_p = ep_v = 0.0
        n_batches = 0

        for x, target_p, target_v in loader:
            x        = x.to(device, non_blocking=True)
            target_p = target_p.to(device, non_blocking=True)
            target_v = target_v.to(device, non_blocking=True)

            pred_p, pred_v       = net(x)
            loss, p_loss, v_loss = alphazero_loss(pred_p, pred_v, target_p, target_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_total  += loss.item()
            ep_p      += p_loss.item()
            ep_v      += v_loss.item()
            n_batches += 1

        if n_batches > 0:
            total_losses.append(ep_total / n_batches)
            policy_losses.append(ep_p / n_batches)
            value_losses.append(ep_v / n_batches)
            log.debug(
                f"  epoch {epoch+1}/{cfg.epochs_per_iteration}  "
                f"loss={ep_total/n_batches:.4f}  "
                f"policy={ep_p/n_batches:.4f}  "
                f"value={ep_v/n_batches:.4f}"
            )

    stats["train_time"]   = time.time() - t0
    stats["loss"]         = total_losses[-1]  if total_losses  else None
    stats["policy_loss"]  = policy_losses[-1] if policy_losses else None
    stats["value_loss"]   = value_losses[-1]  if value_losses  else None

    return stats


# ── メインループ ────────────────────────────────────────────

def main(cfg: Config = DEFAULT) -> None:
    log = setup_main_logger(cfg.log_path)

    device = torch.device(
        cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"
    )
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device}  ({gpu_name})")
    log.info(f"起動: device={device} ({gpu_name})")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)

    net = GomokuNet(
        in_channels=cfg.in_channels,
        filters=cfg.num_filters,
        res_blocks=cfg.num_res_blocks,
        board_size=cfg.board_size,
    ).to(device)

    if Path(cfg.model_path).exists():
        log.info(f"既存モデルを読み込み: {cfg.model_path}")
        net = GomokuNet.load_binary(cfg.model_path, device=device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if Path(cfg.buffer_path).exists():
        buffer = ReplayBuffer.load(cfg.buffer_path)
        log.info(f"バッファ復元: {len(buffer):,} サンプル")
    else:
        buffer = ReplayBuffer(capacity=cfg.replay_buffer_size)

    stats_all = TrainingStats.load(cfg.stats_path)
    if stats_all.total_iterations > 0:
        log.info(f"統計引き継ぎ: {stats_all.total_iterations} イテレーション済み")
        print(stats_all.summary())

    already_done = stats_all.total_iterations
    target_total = already_done + cfg.num_iterations
    remaining    = cfg.num_iterations

    if already_done > 0:
        print(
            f"累計 {already_done} イテレーション完了済み。"
            f"{remaining} イテレーション追加します（目標合計: {target_total}）。"
        )

    # ── イテレーションループ ──────────────────────────────
    for iter_idx in range(1, remaining + 1):
        global_iter = stats_all.total_iterations + 1

        print(
            f"\n=== Iteration {iter_idx}/{remaining}"
            f"  (global #{global_iter} / 目標 {target_total}) ==="
        )
        log.info(f"Iteration開始: {iter_idx}/{remaining} (global #{global_iter})")

        stats = run_iteration(net, optimizer, buffer, cfg, device, iter_idx, global_iter)

        # 統計を更新
        stats_all.total_iterations   += 1
        stats_all.total_games        += cfg.games_per_iteration
        stats_all.total_samples      += stats.get("new_samples", 0)
        stats_all.total_self_play_time += stats.get("self_play_time", 0.0)
        stats_all.total_train_time   += stats.get("train_time", 0.0)
        if stats.get("loss") is not None:
            stats_all.last_loss        = stats["loss"]
            stats_all.last_policy_loss = stats["policy_loss"]
            stats_all.last_value_loss  = stats["value_loss"]
            stats_all.loss_history.append(stats["loss"])

        # ── 標準出力：イテレーション結果サマリー ──────────
        sp_sec    = stats.get("self_play_time", 0)
        tr_sec    = stats.get("train_time", 0)
        buf_size  = stats["buffer_size"]
        new_samp  = stats.get("new_samples", 0)

        print(
            f"  自己対局: {sp_sec:.0f}s  "
            f"学習: {tr_sec:.0f}s  "
            f"buffer: {buf_size:,}  "
            f"new: {new_samp:,}"
        )
        if stats.get("loss") is not None:
            print(
                f"  loss: {stats['loss']:.4f}  "
                f"policy: {stats['policy_loss']:.4f}  "
                f"value: {stats['value_loss']:.4f}"
            )
        print(
            f"  [累計] 対局: {stats_all.total_games:,}局  "
            f"イテレーション: {stats_all.total_iterations}"
        )

        log.info(
            f"Iteration完了: loss={stats.get('loss')}"
            f"  sp={sp_sec:.0f}s  train={tr_sec:.0f}s"
        )

        # モデル・統計・バッファを保存
        net.save_binary(cfg.model_path)
        stats_all.save(cfg.stats_path)
        buffer.save(cfg.buffer_path)

        if stats.get("loss") is not None:
            history = stats_all.loss_history
            if len(history) == 1 or stats["loss"] < min(history[:-1]):
                net.save_binary(cfg.best_model_path)
                print(f"  best_model更新: loss={stats['loss']:.4f}")
                log.info(f"best_model更新: loss={stats['loss']:.4f}")

        if iter_idx % 10 == 0:
            ckpt = Path(cfg.checkpoint_dir) / f"model_iter{stats_all.total_iterations:04d}.bin"
            net.save_binary(str(ckpt))

    print("\n学習完了")
    print(stats_all.summary())
    log.info("学習完了")


if __name__ == "__main__":
    cfg = Config()
    if env_path := os.environ.get("MODEL_OUTPUT"):
        cfg.model_path = env_path
    main(cfg)
