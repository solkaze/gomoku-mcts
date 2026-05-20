"""
ロギング設定モジュール

メインプロセスと子プロセスで共通して使えるロガーを提供する。

multiprocessing + logging の注意点:
  Pythonのloggingはスレッドセーフだがプロセスセーフではない。
  複数プロセスが同じファイルに書くと壊れる可能性があるため、
  子プロセス（inference_server, worker）は専用の setup_worker_logger() を
  呼び出してファイルハンドラを独立して初期化する。
  同一ファイルへの書き込みは os レベルで append モードになるため、
  短いログ行であれば実用上問題ない（Linux のwrite syscall は atomic）。
"""

import logging
import logging.handlers
import sys
from pathlib import Path


LOG_FILE = "/workspace/logs/train.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_main_logger(log_path: str = LOG_FILE) -> logging.Logger:
    """
    メインプロセス用ロガーをセットアップして返す。
    - ファイル: DEBUG以上をローテーションなしで記録
    - 標準出力: WARNING以上のみ（ユーザー向け情報）
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 既存ハンドラをクリア（二重登録防止）
    root.handlers.clear()

    # ── ファイルハンドラ（DEBUG以上、全詳細）──────────────
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(fh)

    # ── 標準出力ハンドラ（INFOのみ、フィルタで制御）────────
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.addFilter(_UserFacingFilter())
    sh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(sh)

    return logging.getLogger("train")


def setup_worker_logger(name: str, log_path: str = LOG_FILE) -> logging.Logger:
    """
    子プロセス（inference_server, worker）用ロガー。
    ファイルにのみ書き、標準出力には出さない。
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False  # ルートロガーには伝搬しない

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(fh)

    return logger


class _UserFacingFilter(logging.Filter):
    """
    標準出力に出すログを絞るフィルタ。
    logger名が "train" のものだけを通す。
    （inference_server や worker の INFO は標準出力に出さない）
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "train"
