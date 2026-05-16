"""
学習ループのメインスクリプト

[自己対局] → [リプレイバッファ追加] → [NN学習] → [モデル保存] を繰り返す。
"""

import os
import time
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


# ── 損失関数 ────────────────────────────────────────────────

def alphazero_loss(
    pred_policy: torch.Tensor,  # (B, 225) softmax済み
    pred_value:  torch.Tensor,  # (B, 1)   tanh済み
    target_policy: torch.Tensor,  # (B, 225) MCTS確率
    target_value:  torch.Tensor,  # (B, 1)   勝敗 [-1, 0, 1]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss = MSE(value) + CrossEntropy(policy)

    pred_policyは既にsoftmax済みなのでlog()してからCEを計算。
    Note: F.cross_entropyではなく自前で書くのは、softmax済みを受けるため。
    """
    eps = 1e-9
    policy_loss = -(target_policy * torch.log(pred_policy + eps)).sum(dim=1).mean()
    value_loss = nn.functional.mse_loss(pred_value, target_value)
    total = policy_loss + value_loss
    return total, policy_loss, value_loss


# ── 1イテレーション ─────────────────────────────────────────

def run_iteration(
    net: GomokuNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: Config,
    device: torch.device,
    iter_idx: int,
) -> dict:
    """
    1イテレーション = (自己対局 → 学習)
    Returns: 統計情報の辞書
    """
    stats = {}

    # ── ① 自己対局（マルチプロセス並列実行）─────────────
    t0 = time.time()
    # 1回の呼び出しで全 games_per_iteration 局を num_workers プロセスで並列実行
    samples = play_games_parallel(net, cfg, device, n_games=cfg.games_per_iteration)
    buffer.add_samples(samples, augment=True)
    new_samples = len(samples)
    elapsed = time.time() - t0
    print(f"  [self-play] {cfg.games_per_iteration} games 完了 ({elapsed:.1f}s)")

    stats["self_play_time"] = time.time() - t0
    stats["new_samples"] = new_samples
    stats["buffer_size"] = len(buffer)

    # 学習可能なサイズに達していなければスキップ
    if len(buffer) < cfg.batch_size:
        print(f"  バッファサイズ不足（{len(buffer)} < {cfg.batch_size}）。学習スキップ。")
        return stats

    # ── ② 学習 ───────────────────────────────────────────
    t0 = time.time()
    dataset = buffer.to_dataset()
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # CUDAテンソルとの相性のため0
        pin_memory=(device.type == "cuda"),
    )

    net.train()
    total_losses = []
    policy_losses = []
    value_losses = []

    for epoch in range(cfg.epochs_per_iteration):
        epoch_total = 0.0
        epoch_p = 0.0
        epoch_v = 0.0
        n_batches = 0

        for x, target_p, target_v in loader:
            x = x.to(device, non_blocking=True)
            target_p = target_p.to(device, non_blocking=True)
            target_v = target_v.to(device, non_blocking=True)

            pred_p, pred_v = net(x)
            loss, p_loss, v_loss = alphazero_loss(pred_p, pred_v, target_p, target_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_p += p_loss.item()
            epoch_v += v_loss.item()
            n_batches += 1

        if n_batches > 0:
            total_losses.append(epoch_total / n_batches)
            policy_losses.append(epoch_p / n_batches)
            value_losses.append(epoch_v / n_batches)

    stats["train_time"] = time.time() - t0
    stats["loss"] = total_losses[-1] if total_losses else None
    stats["policy_loss"] = policy_losses[-1] if policy_losses else None
    stats["value_loss"] = value_losses[-1] if value_losses else None

    return stats


# ── メインループ ────────────────────────────────────────────

def main(cfg: Config = DEFAULT) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 出力ディレクトリ
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)

    # ネットワーク・オプティマイザ
    net = GomokuNet(
        in_channels=cfg.in_channels,
        filters=cfg.num_filters,
        res_blocks=cfg.num_res_blocks,
        board_size=cfg.board_size,
    ).to(device)

    # 既存モデルがあれば再開
    if Path(cfg.model_path).exists():
        print(f"既存モデルを読み込み: {cfg.model_path}")
        net = GomokuNet.load_binary(cfg.model_path, device=device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # バッファの読み込み（既存ファイルがあれば復元）
    from pathlib import Path as _Path
    if _Path(cfg.buffer_path).exists():
        buffer = ReplayBuffer.load(cfg.buffer_path)
        print(f"バッファを復元: {len(buffer):,} サンプル")
    else:
        buffer = ReplayBuffer(capacity=cfg.replay_buffer_size)

    # 学習統計の読み込み（再開時に累計を引き継ぐ）
    stats_all = TrainingStats.load(cfg.stats_path)
    if stats_all.total_iterations > 0:
        print(f"\n統計を引き継ぎ:")
        print(stats_all.summary())

    # ── イテレーションループ ──────────────────────────
    # 累計イテレーション数を目標値として扱う
    # 再実行時は残り分だけ走る
    already_done = stats_all.total_iterations
    target_total = already_done + cfg.num_iterations
    remaining = cfg.num_iterations

    if already_done > 0:
        print(f"\n累計 {already_done} イテレーション完了済み。"
              f"{remaining} イテレーション追加します（目標合計: {target_total}）。")

    for iter_idx in range(1, remaining + 1):
        global_iter = stats_all.total_iterations + 1
        print(f"\n=== Iteration {iter_idx}/{remaining}  (global #{global_iter} / 目標{target_total}) ===")
        stats = run_iteration(net, optimizer, buffer, cfg, device, iter_idx)

        # 統計を更新
        stats_all.total_iterations += 1
        stats_all.total_games += cfg.games_per_iteration
        stats_all.total_samples += stats.get("new_samples", 0)
        stats_all.total_self_play_time += stats.get("self_play_time", 0.0)
        stats_all.total_train_time += stats.get("train_time", 0.0)
        if stats.get("loss") is not None:
            stats_all.last_loss = stats["loss"]
            stats_all.last_policy_loss = stats["policy_loss"]
            stats_all.last_value_loss = stats["value_loss"]
            stats_all.loss_history.append(stats["loss"])

        print(f"  buffer={stats['buffer_size']:6d}  new={stats.get('new_samples', 0):4d}"
              f"  self-play={stats.get('self_play_time', 0):.1f}s")
        if stats.get("loss") is not None:
            print(f"  loss={stats['loss']:.4f}  policy={stats['policy_loss']:.4f}"
                  f"  value={stats['value_loss']:.4f}  train={stats['train_time']:.1f}s")
        print(f"  [累計] 対局={stats_all.total_games:,}局  "
              f"イテレーション={stats_all.total_iterations}")

        # モデル・統計・バッファを保存
        net.save_binary(cfg.model_path)
        stats_all.save(cfg.stats_path)
        buffer.save(cfg.buffer_path)

        # lossが最良を更新したらbest_model.binとして別途保存
        if stats.get("loss") is not None:
            history = stats_all.loss_history
            if len(history) == 1 or stats["loss"] < min(history[:-1]):
                net.save_binary(cfg.best_model_path)
                print(f"  best_model更新: loss={stats['loss']:.4f}")

        if iter_idx % 10 == 0:
            ckpt = Path(cfg.checkpoint_dir) / f"model_iter{stats_all.total_iterations:04d}.bin"
            net.save_binary(str(ckpt))

    print("\n学習完了")
    print(stats_all.summary())


if __name__ == "__main__":
    # 環境変数で出力先を上書き可能（Docker用）
    cfg = Config()
    if env_path := os.environ.get("MODEL_OUTPUT"):
        cfg.model_path = env_path

    main(cfg)
