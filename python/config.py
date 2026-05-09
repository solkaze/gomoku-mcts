"""学習・対局のハイパーパラメータを集約"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── 盤面 ─────────────────────────────────────────
    board_size: int = 15

    # ── ネットワーク ─────────────────────────────────
    in_channels: int = 3
    num_filters: int = 64
    num_res_blocks: int = 5

    # ── MCTS ─────────────────────────────────────────
    n_simulations: int = 400  # 1手あたりの探索回数
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3  # 探索の多様性（ルートにのみ加算）
    dirichlet_eps: float = 0.10  # ノイズの混合率

    # ── 自己対局 ─────────────────────────────────────
    games_per_iteration: int = 50  # 1イテレーションあたりの対局数
    temperature_threshold: int = 15  # この手数まで温度=1（探索）、以降は温度→0
    max_moves: int = 225  # 最大手数

    # ── 学習 ─────────────────────────────────────────
    batch_size: int = 256
    learning_rate: float = 3e-4  # 1e-3 → 3e-4（発散抑制）
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 3  # 5 → 3（過学習抑制）
    replay_buffer_size: int = 200_000  # 50,000 → 200,000

    # ── 学習ループ全体 ───────────────────────────────
    num_iterations: int = 100

    # ── パス ─────────────────────────────────────────
    model_path: str = "/workspace/models/model.bin"
    best_model_path: str = "/workspace/models/best_model.bin"  # 追加
    checkpoint_dir: str = "/workspace/models/checkpoints"
    stats_path: str = "/workspace/models/training_stats.json"
    buffer_path: str = "/workspace/models/replay_buffer.npz"

    # ── デバイス ─────────────────────────────────────
    device: str = "cuda"  # "cpu" / "cuda"


# デフォルトインスタンス（必要に応じてコピーして変更）
DEFAULT = Config()
