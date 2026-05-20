"""学習・対局のハイパーパラメータを集約"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── 盤面 ─────────────────────────────────────────
    board_size: int = 15

    # ── ネットワーク ─────────────────────────────────
    in_channels: int = 3
    num_filters: int = 128
    num_res_blocks: int = 10

    # ── MCTS ─────────────────────────────────────────
    n_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.20

    # ── 自己対局 ─────────────────────────────────────
    games_per_iteration: int = 50

    # ── マルチプロセス並列化 ─────────────────────────
    num_workers: int = 16
    parallel_inner: int = 32
    infer_batch_wait_ms: float = 5.0
    infer_max_batch: int = 1024
    parallel_games: int = 16           # 旧設定（互換用、未使用）
    temperature_threshold: int = 15
    max_moves: int = 225

    # ── 学習 ─────────────────────────────────────────
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 3
    replay_buffer_size: int = 200_000

    # ── 学習ループ全体 ───────────────────────────────
    num_iterations: int = 100

    # ── パス ─────────────────────────────────────────
    model_path: str      = "/workspace/models/model.bin"
    best_model_path: str = "/workspace/models/best_model.bin"
    checkpoint_dir: str  = "/workspace/models/checkpoints"
    stats_path: str      = "/workspace/models/training_stats.json"
    buffer_path: str     = "/workspace/models/replay_buffer.npz"
    log_path: str        = "/workspace/logs/train.log"

    # ── デバイス ─────────────────────────────────────
    device: str = "cuda"


DEFAULT = Config()
