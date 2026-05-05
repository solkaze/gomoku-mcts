"""
学習済みモデル（model.bin）を ONNX 形式に変換するスクリプト

使い方:
  python export_onnx.py                           # デフォルトパス
  python export_onnx.py path/to/model.bin out.onnx
"""

import sys
import torch
from pathlib import Path

from network import GomokuNet, BOARD_SIZE, IN_CHANNELS
from config import DEFAULT


def export(model_path: str, onnx_path: str) -> None:
    print(f"読み込み: {model_path}")
    net = GomokuNet.load_binary(model_path)
    net.eval()

    # ダミー入力（バッチサイズ1の盤面）
    dummy = torch.zeros(1, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"ONNX書き出し: {onnx_path}")
    torch.onnx.export(
        net,
        dummy,
        onnx_path,
        input_names=["board"],          # 入力テンソル名
        output_names=["policy", "value"],
        dynamic_axes={
            "board":  {0: "batch"},     # バッチ次元は可変
            "policy": {0: "batch"},
            "value":  {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,       # 定数畳み込みで最適化
    )

    # 検証: 元モデルとONNX変換後の出力が一致するか
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    test_input = torch.randn(1, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE).numpy()
    onnx_outs = sess.run(None, {"board": test_input})

    with torch.no_grad():
        torch_outs = net(torch.from_numpy(test_input))

    p_diff = abs(onnx_outs[0] - torch_outs[0].numpy()).max()
    v_diff = abs(onnx_outs[1] - torch_outs[1].numpy()).max()
    print(f"  policy max diff: {p_diff:.2e}")
    print(f"  value  max diff: {v_diff:.2e}")
    if p_diff < 1e-4 and v_diff < 1e-4:
        print("変換成功（PyTorchと数値が一致）")
    else:
        print("警告: 数値が大きくズレています")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        onnx_path = sys.argv[2]
    else:
        # デフォルト: best_model.bin → best_model.onnx
        model_path = DEFAULT.best_model_path
        onnx_path = model_path.replace(".bin", ".onnx")

    if not Path(model_path).exists():
        # best が無ければ通常の model.bin を使う
        fallback = DEFAULT.model_path
        if Path(fallback).exists():
            print(f"{model_path} が見つかりません。代わりに {fallback} を使用")
            model_path = fallback
            onnx_path = fallback.replace(".bin", ".onnx")
        else:
            print(f"エラー: モデルファイルが見つかりません: {model_path}")
            sys.exit(1)

    export(model_path, onnx_path)
