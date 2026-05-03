# ── ベースイメージ ───────────────────────────────────────────
# CUDA 13.2.1 + cuDNN（学習時の最適化に必須）
# RTX 5070 Ti (Blackwell/sm_120) 対応
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

# ── システム依存 ─────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.12 /usr/bin/python

# ── venv を /opt/venv に作成 ─────────────────────────────────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── PyTorch ─────────────────────────────────────────────────
# 安定版は cu130 まで。cu132 ナイトリーは 2.12 リリース後に切り替え予定
# （PyTorch 開発ML: https://dev-discuss.pytorch.org/t/3337）
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu130

# ── その他依存 ───────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── 作業ディレクトリ ─────────────────────────────────────────
WORKDIR /workspace

VOLUME ["/workspace/models"]

# マルチプロセス学習のために共有メモリを増やす
# （docker-compose 側で shm_size を設定するが念のため明示）
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
