import os
import argparse
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torchattacks
import numpy as np
import torch.nn as _nn

# -------------------- CIFAR10 包装类 --------------------
class IndexedCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

# -------------------- 自定义 VGG_CIFAR --------------------
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
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------------------- 模型加载 --------------------
def load_model(arch, path, device):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 10)
    elif arch == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 10)
    elif arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, 10)
    elif arch == "wideresnet":
        m = models.wide_resnet50_2(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 10)
    elif arch == "vgg_cifar":
        m = VGG_CIFAR(num_classes=10)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    state = torch.load(path, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m

# -------------------- 主函数 --------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.247,0.243,0.261))
    ])
    dataset = IndexedCIFAR10(root='data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # 加载代理模型
    proxy_path = os.path.join("outputs/proxies", args.proxy_for_craft)
    if not os.path.isfile(proxy_path):
        raise FileNotFoundError(f"Proxy model not found: {proxy_path}")
    proxy = load_model(args.proxy_arch, proxy_path, device)

    # 攻击器：固定 PGD 参数
    eps, steps, alpha = 8/255, 10, 2/255
    attack = torchattacks.PGD(proxy, eps=eps, alpha=alpha, steps=steps)

    # 加载打分结果
    score_data = np.load(args.scores, allow_pickle=True).item()
    scores = score_data["scores"] if isinstance(score_data, dict) else score_data
    n_total = len(scores)

    # 定义 Top-K 区间 & 混合比例
    intervals = [(0.0,0.1), (0.1,0.2), (0.2,0.3), (0.3,0.4), (0.4,0.5)]
    ratios = [0.25, 0.5, 1.0, 2.0, 4.0]

    # 按分数排序
    sorted_idx = np.argsort(-scores.max(axis=1))  # 用 max prob 打分
    print(f"Total samples ranked: {len(sorted_idx)}")

    # 统一样本数：取最小区间数
    min_size = min([int((hi-lo)*n_total) for lo,hi in intervals])
    print(f"Each interval will use {min_size} samples (equal size mode).")

    # 遍历每个区间和比例
    for lo, hi in intervals:
        lo_idx = int(lo * n_total)
        hi_idx = int(hi * n_total)
        sel_idx = sorted_idx[lo_idx:hi_idx]
        if len(sel_idx) > min_size:
            sel_idx = np.random.choice(sel_idx, min_size, replace=False)
        selected_indices = set(sel_idx.tolist())
        print(f"Interval [{lo:.1f},{hi:.1f}] -> {len(selected_indices)} samples")

        # 生成对抗样本
        adv_all, lbl_all = [], []
        for images, labels, indices in loader:
            mask = [i.item() in selected_indices for i in indices]
            if not any(mask):
                continue
            sel_pos = [i for i,v in enumerate(mask) if v]
            images = images[sel_pos].to(device)
            labels = labels[sel_pos].to(device)
            adv = attack(images, labels)
            adv_all.append(adv.cpu())
            lbl_all.append(labels.cpu())

        adv_data = torch.cat(adv_all, dim=0)
        adv_labels = torch.cat(lbl_all, dim=0)

        # 遍历不同 clean 比例
        for r in ratios:
            n_clean = int(len(adv_data) * r)

            # 随机采样 clean 样本
            clean_set = datasets.CIFAR10(root='data', train=True, download=True,
                                         transform=transforms.ToTensor())
            clean_indices = np.random.choice(len(clean_set), n_clean, replace=False)
            clean_imgs, clean_lbls = [], []
            for i in clean_indices:
                img, lbl = clean_set[i]
                img = transforms.Normalize((0.4914,0.4822,0.4465),
                                           (0.247,0.243,0.261))(img)
                clean_imgs.append(img)
                clean_lbls.append(lbl)
            clean_imgs = torch.stack(clean_imgs, dim=0)
            clean_lbls = torch.tensor(clean_lbls, dtype=torch.long)

            # 合并数据
            final_data = torch.cat([adv_data, clean_imgs], dim=0)
            final_labels = torch.cat([adv_labels, clean_lbls], dim=0)

            # 保存路径
            out_dir = os.path.join(args.out,
                                   f"interval_{int(lo*100)}_{int(hi*100)}",
                                   f"ratio_{r}")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "adv_set.pt")

            torch.save({"data": final_data, "labels": final_labels}, save_path)
            print(f"  -> Saved {save_path}, size={len(final_data)}")

# -------------------- 命令行接口 --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=str, required=True,
                   help="Path to scores.npy from compute_scores.py")
    p.add_argument("--proxy-for-craft", type=str, required=True,
                   help="Name of proxy checkpoint under outputs/proxies/")
    p.add_argument("--proxy-arch", type=str, required=True,
                   choices=["resnet18","resnet34","densenet121","wideresnet","vgg_cifar"],
                   help="Architecture of the proxy model")
    p.add_argument("--out", type=str, default="outputs/adv_sets_batch")
    args = p.parse_args()
    main(args)
