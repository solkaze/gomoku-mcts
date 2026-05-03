"""学習統計を表示する確認用スクリプト

使い方:
  python show_stats.py [stats_path]
"""

import sys

from config import DEFAULT
from stats import TrainingStats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT.stats_path
    stats = TrainingStats.load(path)

    if stats.total_iterations == 0:
        print(f"統計ファイルが見つかりません: {path}")
        print("（まだ学習が始まっていない可能性があります）")
        return

    print(f"統計ファイル: {path}\n")
    print(stats.summary())

    # loss推移（直近10回）
    if stats.loss_history:
        recent = stats.loss_history[-10:]
        print("\n直近のloss推移:")
        for i, l in enumerate(recent, start=len(stats.loss_history) - len(recent) + 1):
            print(f"  iter {i:4d}: {l:.4f}")


if __name__ == "__main__":
    main()
