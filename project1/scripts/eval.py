import os, argparse, csv, glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torchattacks

# ---------------- CIFAR-10 均值方差 ----------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2430, 0.2610)

# ---------------- VGG_CIFAR 定义 ----------------
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
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------- Normalize Wrapper ----------------
class NormalizeWrapper(nn.Module):
    """确保输入先归一化"""
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.model = model
        mean_t = torch.tensor(mean).view(1,3,1,1)
        std_t  = torch.tensor(std).view(1,3,1,1)
        self.register_buffer('mean', mean_t)
        self.register_buffer('std', std_t)
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

# ---------------- 模型加载 ----------------
def load_model_ckpt(ckpt_path, arch, device):
    if arch == "resnet18":
        base = models.resnet18(pretrained=False)
        base.fc = nn.Linear(base.fc.in_features, 10)
    elif arch == "vgg_cifar":
        base = VGG_CIFAR(num_classes=10)
    else:
        raise ValueError(f"Unsupported architecture {arch}")

    state = torch.load(ckpt_path, map_location="cpu")
    base.load_state_dict(state)
    base.to(device).eval()
    return base

# ---------------- 数据加载 ----------------
def get_test_loader(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])  # 保持 [0,1]
    testset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------- Clean 测试 ----------------
def eval_clean(model_wrapper, device, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model_wrapper(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

# ---------------- PGD 测试 ----------------
def eval_pgd(model_wrapper, device, test_loader, eps=8/255, alpha=2/255, steps=10):
    atk = torchattacks.PGD(model_wrapper, eps=eps, alpha=alpha, steps=steps)
    correct, total = 0, 0
    for x,y in test_loader:
        x, y = x.to(device), y.to(device)
        adv = atk(x, y)
        out = model_wrapper(adv)
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    return correct/total

# ---------------- 主函数 ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader(batch_size=args.batch_size)

    # 找到所有 checkpoint
    if os.path.isdir(args.model_dir):
        ckpts = sorted(glob.glob(os.path.join(args.model_dir, "*.pth")))
    else:
        raise ValueError("请传入模型文件夹 (--model-dir)")

    results = []
    for ck in ckpts:
        print(f"\nEvaluating {ck} ...")
        model = load_model_ckpt(ck, args.arch, device)
        model_wrapper = NormalizeWrapper(model).to(device)

        # 1) Clean acc
        clean_acc = eval_clean(model_wrapper, device, test_loader)
        print("Clean acc:", clean_acc)

        # 2) PGD acc
        pgd_results = {}
        for eps in args.pgd_eps_list:
            acc = eval_pgd(model_wrapper, device, test_loader, eps=eps, alpha=min(eps/4, 2/255), steps=args.pgd_steps)
            print(f"PGD eps={eps:.5f} acc={acc:.4f}")
            pgd_results[f"pgd_eps_{eps:.5f}"] = acc

        # 收集结果
        res = {
            "arch": args.arch,
            "ckpt": os.path.basename(ck),
            "clean_acc": clean_acc,
        }
        res.update(pgd_results)
        results.append(res)

        # 保存到 CSV
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(res)
        print(f"Appended results to {args.out_csv}")

    print("\nAll done.")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=["resnet18", "vgg_cifar"],
                        help="模型架构")
    parser.add_argument('--model-dir', type=str, required=True,
                        help="存放 .pth 文件的文件夹")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument('--pgd-eps-list', nargs='+', type=float, default=[2/255, 4/255, 8/255])
    parser.add_argument('--out-csv', type=str, default='outputs/eval_results.csv')
    args = parser.parse_args()
    main(args)
