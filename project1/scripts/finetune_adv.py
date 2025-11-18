import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2430, 0.2610)

# -------------------- VGG_CIFAR --------------------
class VGG_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------- 加载数据集 --------------------
def load_mixed_dataset(path):
    d = torch.load(path, map_location='cpu')
    data = d['data']
    labels = d['labels'].long()
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    data = torch.stack([norm(x) for x in data], dim=0)
    return TensorDataset(data, labels)

# -------------------- 构建模型 --------------------
def build_model(arch, num_classes=10, device=None, init_from=None):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "vgg_cifar":
        model = VGG_CIFAR(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    if init_from:
        state = torch.load(init_from, map_location='cpu')
        model.load_state_dict(state)
    if device:
        model = model.to(device)
    return model

# -------------------- 单次训练 --------------------
def train_single(args, adv_file, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    dataset = load_mixed_dataset(adv_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Loaded dataset {adv_file}: {len(dataset)} samples")

    model = build_model(args.arch, device=device, init_from=args.init_model)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        running_loss = 0.0
        total = 0; correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            total += yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
        scheduler.step()
        train_acc = correct/total if total>0 else 0.0
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={running_loss/total:.4f} train_acc={train_acc:.4f}")

        if args.save_every > 0 and (epoch+1) % args.save_every == 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            tmp_path = out_path.replace('.pth', f'.epoch{epoch+1}.pth') if out_path.endswith('.pth') else os.path.join(out_path, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), tmp_path)
            print(f"Saved checkpoint: {tmp_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print("Finished training. Saved model to", out_path)

# -------------------- 批量训练 --------------------
def train_batch(args, adv_files, out_dir):
    for adv_file in adv_files:
        base_name = os.path.splitext(os.path.basename(adv_file))[0]
        out_path = os.path.join(out_dir, f"{base_name}_{args.arch}.pth")
        print(f"\n=== Training on {adv_file}, output -> {out_path} ===")
        train_single(args, adv_file, out_path)

# -------------------- 命令行接口 --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['resnet18','vgg_cifar'])
    parser.add_argument('--init-model', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--step-size', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    # 功能选择：单个 vs 批量
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--adv-file', type=str, help='单个混合数据集 (.pt)')
    group.add_argument('--adv-dir', type=str, help='批量混合数据集目录 (.pt 文件)')

    parser.add_argument('--out', type=str, required=True, help='输出路径或目录')
    args = parser.parse_args()

    if args.adv_file:
        # 单个混合训练
        train_single(args, args.adv_file, args.out)
    elif args.adv_dir:
        # 批量训练
        adv_files = [os.path.join(args.adv_dir, f) for f in os.listdir(args.adv_dir) if f.endswith('.pt')]
        train_batch(args, adv_files, args.out)
