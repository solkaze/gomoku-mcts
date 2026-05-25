# gomoku-mcts プロジェクト概要

## 構成

AlphaZero方式の五目並べAI（Python/PyTorch実装）。

```
python/
  network.py          # GomokuNet: Policy + Value 2ヘッドCNN
  mcts.py             # MCTS本体（単体デバッグ用）
  worker_mcts.py      # Virtual Loss付きMCTS（ワーカー用）
  worker.py           # 自己対局ワーカープロセス
  self_play.py        # マルチプロセス自己対局管理
  inference_server.py # GPU推論サーバー（バッチ処理）
  dataset.py          # ReplayBuffer + データ拡張（8対称）
  train.py            # 学習ループ本体
  config.py           # ハイパーパラメータ

rust/
  src/board.rs        # 盤面（15×15、勝利判定）
  src/forbidden.rs    # 禁じ手判定（三三・四四・長連）
  src/ai/             # 推論・対局（ONNX利用予定）
```

## ゲームルール

- 盤面: 15×15
- 石の表現: 黒=1（先手）, 白=2（後手）, 空=0
- 勝利: 5連以上（長連も勝利扱い ※禁じ手は先手のみ）
- 禁じ手: 先手（黒）のみ 三三・四四・長連が適用

## NN入力テンソル

shape: `(batch, 3, 15, 15)` / dtype: float32

| チャンネル | 内容 |
|-----------|------|
| ch0 | 現在の手番プレイヤーの石 |
| ch1 | 相手の石 |
| ch2 | 手番定数面（先手=1.0, 後手=0.0） |

## 価値の符号規約

**「現在の手番プレイヤーから見た価値」** を使う。

- `+1.0` = このノードのto_playが勝つ
- `-1.0` = このノードのto_playが負ける
- `0.0`  = 引き分け

backpropでは親ノードに上がるたびに符号を反転する。

## MCTSの実装方針

- Virtual Loss使用（`worker_mcts.py`）
- Dirichletノイズはルートにのみ付加
- 推論はGPU専属の `inference_server` に集約しバッチ処理
- `board_to_tensor` で視点変換（常に「自分の石がch0」）

## 既知のバグ（修正対象）

本プロジェクトには学習が全く進まない原因となるバグが複数存在する。
詳細は投入するプロンプトを参照。
