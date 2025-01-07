import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import sys

sys.path.append('.')
#from Lenet5 import Lenet5
from resnet import ResNet18


def main():
    batchz = 128  # 尝试缩小批次大小，避免内存溢出
    # CIFAR10 数据集下载路径可以尝试改成默认路径
    cifar_train = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchz, shuffle=True)

    cifar_test = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchz, shuffle=True)

    device = torch.device('cpu')  # 使用CPU
    # 确保 ResNet18 模型定义正确
    model = ResNet18().to(device)
    crition = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        model.train()
        for batch, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = crition(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
            acc = total_correct / total_num
            print(f"Epoch {epoch}, test acc: {acc:.4f}")


if __name__ == '__main__':
    main()
