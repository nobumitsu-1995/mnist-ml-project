# mnist-ml-project

PyTorch を用いて MNIST（手書き数字）データセットを分類する、畳み込みニューラルネットワーク（CNN）と全結合ニューラルネットワーク（MLP）の2種類のモデルを学習・評価できる学習用プロジェクトです。

## 概要

- **タスク**: MNIST 手書き数字（0〜9）の10クラス分類
- **モデル構成**: 2種類のモデルを実装（`main.py` 内で切り替え可能）

### 1. `SimpleNet`（全結合ネットワーク / MLP）

3層の全結合ネットワーク。

| 層 | 構成 |
| --- | --- |
| 入力 | 784（28×28 ピクセルを平坦化） |
| 隠れ層1 | 784 → 128（ReLU） |
| 隠れ層2 | 128 → 64（ReLU） |
| 出力層 | 64 → 10 |

### 2. `ConvNet`（畳み込みネットワーク / CNN）※デフォルト

畳み込み層 + プーリング層 + 全結合層で構成されたネットワーク。

| 層 | 構成 |
| --- | --- |
| 畳み込み層1 | 入力 1ch（白黒）→ 出力 32ch、3×3 カーネル、padding=1（ReLU） |
| プーリング1 | MaxPool 2×2（28×28 → 14×14） |
| 畳み込み層2 | 入力 32ch → 出力 64ch、3×3 カーネル、padding=1（ReLU） |
| プーリング2 | MaxPool 2×2（14×14 → 7×7） |
| 全結合層1 | 3136（64×7×7）→ 128（ReLU） |
| 出力層 | 128 → 10 |

### 学習設定

- 最適化アルゴリズム: Adam
- 損失関数: CrossEntropyLoss
- 学習率: 0.001
- エポック数: 10
- バッチサイズ: 64

### デバイス対応

CUDA（NVIDIA GPU）／ MPS（Apple Silicon）／ CPU を自動判定して使用します。

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `main.py` | モデル定義（`SimpleNet`、`ConvNet`）と学習スクリプト。学習結果を `mnist_model.pth` に保存します。 |
| `predict.py` | 保存済みモデルを読み込み、MNIST テストデータ（10,000件）で正解率を評価します。 |
| `requirements.txt` | 依存ライブラリのバージョン定義 |
| `data/` | MNIST データセットのダウンロード先（git管理対象外） |
| `mnist_model.pth` | 学習済みモデルの重み（git管理対象外） |

## セットアップ

### 1. 仮想環境の作成と有効化

```bash
python -m venv .venv
source .venv/bin/activate  # Windows の場合: .venv\Scripts\activate
```

### 2. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

## 使い方

### 1. モデルの学習

```bash
python main.py
```

- MNIST データセットが `./data` 配下に自動ダウンロードされます。
- 学習が完了すると、重みが `mnist_model.pth` に保存されます。
- 100バッチごとに学習中の損失（Loss）が表示されます。

### 2. モデルの評価

```bash
python predict.py
```

- `mnist_model.pth` を読み込み、MNIST テストデータでの正解率を表示します。

実行例:

```
==============================
テストデータ件数: 10000
正解数: 9900
正解率: 99.00%
==============================
```

### モデルの切り替え方

デフォルトでは `ConvNet`（CNN）が使用されます。`SimpleNet`（MLP）を使う場合は、`main.py` と `predict.py` のモデル生成箇所のコメントを切り替えてください。

```python
# main.py / predict.py
model = SimpleNet().to(device)
# model = ConvNet().to(device)
```

※ 学習時と評価時は同じモデルクラスを指定してください（重みの構造が一致しないと読み込みに失敗します）。
