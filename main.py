import torch
from torchvision import datasets, transforms

# ==========================================
# 使用デバイスの判定
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda") # windows + GPU用設定
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Mac用設定
else:
    device = torch.device("cpu") # その他用の設定

print(f"using device: {device}")

# ==========================================
# 3層の全結合ネットワーク構築
# 隠れ層1(784 → 128) → 隠れ層2(128 → 64) → 出力層(64 → 10)
# ==========================================
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.flatten(x) # ここでピクセルデータをテンソル変換し平坦化
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# ==========================================
# 畳み込み層 + 全結合ネットワーク構築
# 畳み込み層1 (1枚 → 32部品) → 圧縮1 (28x28 → 14x14 ※面積1/4) → 畳み込み層2 (32部品 → 64特徴) → 圧縮2 (14x14 → 7x7 ※面積1/4) → 隠れ層1 (3136 → 128) → 隠れ層2 (128 → 64) → 出力層 (64 → 10)
# ==========================================
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1. 畳み込み層: 入力1ch(白黒), 出力32ch, 3x3のフィルタ
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 2. 畳み込み層: 入力32ch, 出力64ch, 3x3のフィルタ
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 3. プーリング層: 2x2の範囲で最大値を取り出し、サイズを半分に圧縮
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        
        # 最終的な全結合層への入力サイズ計算: 
        # 28x28 -> (conv1) -> 28x28 -> (pool) -> 14x14 
        # -> (conv2) -> 14x14 -> (pool) -> 7x7
        # 64枚の7x7画像 = 64 * 7 * 7 = 3136
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # [Batch, 1, 28, 28] -> Conv1 -> ReLU -> Pool -> [Batch, 32, 14, 14]
        x = self.pool(self.relu(self.conv1(x)))
        # [Batch, 32, 14, 14] -> Conv2 -> ReLU -> Pool -> [Batch, 64, 7, 7]
        x = self.pool(self.relu(self.conv2(x)))
        
        # 全結合層に渡すために1次元にフラット化
        x = x.view(-1, 64 * 7 * 7)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# メインの処理関数
# ==========================================
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
LEARNING_RATE = 0.001
EPOCHS=10
BATCH_SIZE=64


def main():
    # model = SimpleNet().to(device)
    model = ConvNet().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(), # 0-255(Int) -> 0.0-1.0(Float)への変換
        transforms.Normalize(MNIST_MEAN, MNIST_STD) # MNISTデータセットの偏差計算による標準化
    ])

    # DataLoaderを使用することでメモリ効率やシャッフル処理(AIが「出現順序」という無意味なノイズを学習してしまうのを防ぐ処理)を気にせず、綺麗なバッチデータを受けとる
    train_dataset = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Adamアルゴリズムによる「重み（Weight）」と「バイアス（Bias）」の更新強度の設定
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 評価関数（物差し）を生成
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_dataset):
            data, target = data.to(device), target.to(device)
            # 1. 過去の記憶（前回の勾配）を消去（リセット）
            optimizer.zero_grad() 
            # 2. 現状の重みに基づく予測値（推論結果）を最終出力として取得
            output = model(data)             
            # 3. 交差エントロピーによる期待値と実際の値の乖離をスコア化
            loss = criterion(output, target)            
            # 4. 誤差から「どの変数をどっちに動かすべきか」を計算
            loss.backward()            
            # 5. 実際に Adam のアルゴリズムに従って重みを書き換える（更新実行）
            optimizer.step()

            if batch_idx % 100 == 0:  # 100バッチごとに進捗を表示
                print(f'Train Batch: {batch_idx} \tLoss: {loss.item():.6f}')
    
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Saved model to mnist_model.pth")

if __name__ == "__main__":
    main()