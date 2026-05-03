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
    n_simulations: int = 200  # 1手あたりの探索回数
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3  # 探索の多様性（ルートにのみ加算）
    dirichlet_eps: float = 0.25  # ノイズの混合率

    # ── 自己対局 ─────────────────────────────────────
    games_per_iteration: int = 50  # 1イテレーションあたりの対局数
    temperature_threshold: int = 15  # この手数まで温度=1（探索)、以降は温度→0
    max_moves: int = 225  # 最大手数（盤面が埋まる前にcut）

    # ── 学習 ─────────────────────────────────────────
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 5  # 1イテレーションあたりの学習エポック数
    replay_buffer_size: int = 200_000  # 直近の局面数（古いものから捨てる)

    # ── 学習ループ全体 ───────────────────────────────
    num_iterations: int = 100  # イテレーション総数

    # ── パス ─────────────────────────────────────────
    model_path: str = "/workspace/models/model.bin"
    checkpoint_dir: str = "/workspace/models/checkpoints"
    stats_path: str = "/workspace/models/training_stats.json"

    # ── デバイス ─────────────────────────────────────
    device: str = "cuda"  # "cpu" / "cuda"


# デフォルトインスタンス（必要に応じてコピーして変更）
DEFAULT = Config()
