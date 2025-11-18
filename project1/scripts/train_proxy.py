"""
示例：
  python scripts/train_proxy.py --arch resnet18
  python scripts/train_proxy.py --arch resnet18,vgg16,wideresnet --epochs 30

输出：
  outputs/proxies/resnet18.pth
  outputs/proxies/vgg16.pth
  outputs/proxies/wideresnet.pth
"""

import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# CIFAR 均值/方差
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2430, 0.2610)


def get_dataloaders(batch_size=128, num_workers=0):
    """返回 train_loader, test_loader
    num_workers: Windows 下建议 0，Linux 下可设为 >0 提速。
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    train = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    test  = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# 替换 train_proxy.py 中的 build_model 函数为下面内容

def build_model(arch="resnet18", num_classes=10, device=None):
    """
    支持架构:
      - resnet18
      - resnet34
      - vgg16            (原始 ImageNet 版，不推荐用于 32x32)
      - vgg_cifar        (改造版 VGG，适配 CIFAR-10)
      - wideresnet       (wide_resnet50_2)
      - densenet121
      - mobilenet_v2
    """
    if arch == "resnet18":
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "resnet34":
        m = models.resnet34(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "vgg16":
        # 原始 vgg16（ImageNet），不太适合 32x32，但保留选项
        m = models.vgg16(pretrained=False)
        m.classifier[6] = nn.Linear(4096, num_classes)

    elif arch == "vgg_cifar":
        # 一个简单的 VGG 风格网络，适配 CIFAR-10（32x32）
        # 我们构造一个小版 VGG：只用 3x3 conv，较少 pool，最后全局池化替代大 fc
        from torch import nn as _nn
        class VGG_CIFAR(_nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = _nn.Sequential(
                    _nn.Conv2d(3, 64, kernel_size=3, padding=1), _nn.ReLU(inplace=True),
                    _nn.Conv2d(64, 64, kernel_size=3, padding=1), _nn.ReLU(inplace=True),
                    _nn.MaxPool2d(2,2),  # 16x16

                    _nn.Conv2d(64, 128, kernel_size=3, padding=1), _nn.ReLU(inplace=True),
                    _nn.Conv2d(128, 128, kernel_size=3, padding=1), _nn.ReLU(inplace=True),
                    _nn.MaxPool2d(2,2),  # 8x8

                    _nn.Conv2d(128, 256, kernel_size=3, padding=1), _nn.ReLU(inplace=True),
                    _nn.Conv2d(256, 256, kernel_size=3, padding=1), _nn.ReLU(inplace=True),
                    _nn.MaxPool2d(2,2),  # 4x4
                )
                self.classifier = _nn.Sequential(
                    _nn.AdaptiveAvgPool2d((1,1)),  # 256 x 1 x 1
                    _nn.Flatten(),
                    _nn.Linear(256, 256),
                    _nn.ReLU(inplace=True),
                    _nn.Dropout(0.5),
                    _nn.Linear(256, num_classes)
                )
            def forward(self, x): return self.classifier(self.features(x))
        m = VGG_CIFAR(num_classes=num_classes)

    elif arch == "wideresnet":
        m = models.wide_resnet50_2(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "densenet121":
        m = models.densenet121(pretrained=False)
        # densenet classifier is m.classifier
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown arch {arch}")

    if device:
        m = m.to(device)
    return m


def train_one(model, train_loader, test_loader, args, arch_name):
    """单个模型的训练循环（标准训练 + 在 test 上选最优模型保存）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = os.path.join(args.out_dir, f"{arch_name}60.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0; correct = 0; running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0

        # eval
        model.eval()
        tot = 0; corr = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device); y = y.to(device)
                out = model(x)
                tot += y.size(0)
                corr += (out.argmax(1) == y).sum().item()
        test_acc = corr / tot if tot > 0 else 0.0

        print(f"[{arch_name}] Epoch {epoch}/{args.epochs}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  loss={running_loss/total:.4f}")

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), out_path)
            print(f"[{arch_name}] Saved best model to {out_path} (acc={best_acc:.4f})")

    print(f"[{arch_name}] Finished. Best test acc: {best_acc:.4f}")
    return out_path, best_acc


def main(args):
    # 支持 --arch resnet18 或 --arch resnet18,vgg16,wideresnet
    archs = []
    if ',' in args.arch:
        archs = [a.strip() for a in args.arch.split(',') if a.strip()]
    else:
        archs = [args.arch]

    # 构建 dataloaders（共用）
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    results = []
    for arch in archs:
        print(f"=== Start training arch: {arch} ===")
        model = build_model(arch=arch, device=None)  # model 放到 train_one 内的 device
        out_path, best_acc = train_one(model, train_loader, test_loader, args, arch)
        results.append((arch, out_path, best_acc))

    print("All done. Summary:")
    for arch, path, acc in results:
        print(f"  - {arch}: best_model={path}, best_acc={acc:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train proxy models (resnet18/vgg16/wideresnet)")
    p.add_argument('--arch', type=str, default='resnet18',
                   help="模型架构名称，单个或逗号分隔多个（resnet18,vgg16,wideresnet）")
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--step-size', type=int, default=40, help='LR scheduler step size')
    p.add_argument('--gamma', type=float, default=0.1, help='LR scheduler gamma')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers (Windows 推荐0)')
    p.add_argument('--out-dir', type=str, default='outputs/proxies', help='代理模型保存目录')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
