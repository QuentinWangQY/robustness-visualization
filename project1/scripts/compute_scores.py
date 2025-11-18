import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------- CIFAR 预处理 ----------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2430, 0.2610)

# ---------------- VGG_CIFAR 模型定义 ----------------
class VGG_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # 256x1x1
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------- 模型构建函数 ----------------
def build_model(arch, num_classes=10, ckpt_path=None, device="cpu"):
    if arch == "resnet18":
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "resnet34":
        m = models.resnet34(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "wideresnet":
        m = models.wide_resnet50_2(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "densenet121":
        m = models.densenet121(pretrained=False)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif arch == "vgg_cifar":
        m = VGG_CIFAR(num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported arch {arch}")

    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

# ---------------- 融合函数 ----------------
def fuse_logits(logits_list, method="mean"):
    """融合多个模型的输出 logits"""
    if method == "mean":
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)
    elif method == "max":
        return torch.max(torch.stack(logits_list, dim=0), dim=0).values
    elif method == "vote":
        preds = [logits.argmax(dim=1) for logits in logits_list]
        preds = torch.stack(preds, dim=0)  # [num_models, batch_size]
        # 投票：每列取众数
        fused = []
        for i in range(preds.size(1)):
            vals, counts = preds[:, i].unique(return_counts=True)
            fused.append(vals[counts.argmax()])
        fused = torch.tensor(fused, device=preds.device)
        # 转成 one-hot 形式（便于 softmax 一致）
        one_hot = torch.zeros(preds.size(1), logits_list[0].size(1), device=preds.device)
        one_hot[torch.arange(preds.size(1)), fused] = 1.0
        return one_hot
    else:
        raise ValueError(f"Unknown fusion method {method}")

# ---------------- 主函数 ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CIFAR-10 测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 加载所有模型
    model_infos = []
    for entry in args.models:
        arch, ckpt = entry.split("::")
        model = build_model(arch, num_classes=10, ckpt_path=ckpt, device=device)
        model_infos.append((arch, ckpt, model))
        print(f"[Loaded] {arch} from {ckpt}")

    all_scores = []
    all_labels = []

    # 遍历测试集
    for x, y in tqdm(test_loader, desc="Scoring"):
        x, y = x.to(device), y.to(device)

        # 收集所有模型的输出 logits
        logits_list = []
        for arch, ckpt, model in model_infos:
            with torch.no_grad():
                logits = model(x)
                logits_list.append(logits)

        # 模型融合
        fused_logits = fuse_logits(logits_list, method=args.fusion)
        scores = torch.softmax(fused_logits, dim=1)

        all_scores.append(scores.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, {"scores": all_scores, "labels": all_labels})
    print(f"[Done] Saved scores to {args.out}")

# ---------------- 命令行入口 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="格式: arch::ckpt_path，例如 'resnet18::outputs/models/resnet18_1.pth'")
    parser.add_argument("--out", type=str, default="outputs/scores.npy")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fusion", type=str, default="mean",
                        choices=["mean", "max", "vote"],
                        help="模型融合方式: mean=平均, max=取最大, vote=投票")
    args = parser.parse_args()
    main(args)
