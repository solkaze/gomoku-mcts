"""
学習統計の保存・読み込み
学習を中断・再開しても累計が引き継がれる。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingStats:
    total_iterations: int = 0
    total_games: int = 0  # 自己対局の累計局数
    total_samples: int = 0  # 拡張前のサンプル累計（局面数）
    total_train_time: float = 0.0  # 学習にかかった累計秒数
    total_self_play_time: float = 0.0
    last_loss: Optional[float] = None
    last_policy_loss: Optional[float] = None
    last_value_loss: Optional[float] = None
    loss_history: list[float] = field(default_factory=list)  # 各イテレーション末のloss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingStats":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        return (
            f"累計イテレーション: {self.total_iterations}\n"
            f"累計自己対局数:     {self.total_games:,} 局\n"
            f"累計サンプル数:     {self.total_samples:,} 局面\n"
            f"累計自己対局時間:   {self.total_self_play_time / 3600:.2f} 時間\n"
            f"累計学習時間:       {self.total_train_time / 3600:.2f} 時間\n"
            f"最新 loss:          {self.last_loss:.4f}"
            if self.last_loss
            else "最新 loss: -"
        )


if __name__ == "__main__":
    # 動作確認
    s = TrainingStats()
    s.total_iterations = 5
    s.total_games = 250
    s.total_samples = 12500
    s.last_loss = 2.34
    s.save("/tmp/test_stats.json")

    s2 = TrainingStats.load("/tmp/test_stats.json")
    print(s2.summary())
