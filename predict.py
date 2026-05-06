import torch
from torchvision import datasets, transforms
from main import SimpleNet, ConvNet, device, MNIST_MEAN, MNIST_STD, BATCH_SIZE


def evaluate(model_path):
    # モデルの読み込み
    # model = SimpleNet().to(device)
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(), # 0-255(Int) -> 0.0-1.0(Float)への変換
        transforms.Normalize(MNIST_MEAN, MNIST_STD) # MNISTデータセットの偏差計算による標準化
    ])

    # MNISTのテスト用データセット（10,000件）を読み込み
    test_dataset = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dataset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = output.argmax(dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"==============================")
    print(f"テストデータ件数: {total}")
    print(f"正解数: {correct}")
    print(f"正解率: {accuracy:.2f}%")
    print(f"==============================")


if __name__ == "__main__":
    evaluate("mnist_model.pth")
