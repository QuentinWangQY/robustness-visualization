import os
import argparse
import subprocess

def main(args):
    # 模型文件夹（这里你需要修改成自己的实际路径）
    resnet_dirs = [
        "outputs/models/resnet18_0_10",
        "outputs/models/resnet18_10_20",
        "outputs/models/resnet18_20_30",
        "outputs/models/resnet18_30_40",
        "outputs/models/resnet18_40_50",
    ]
    vgg_dirs = [
        "outputs/models/vgg_cifar_0_10",
        "outputs/models/vgg_cifar_10_20",
        "outputs/models/vgg_cifar_20_30",
        "outputs/models/vgg_cifar_30_40",
        "outputs/models/vgg_cifar_40_50",
    ]

    # 输出结果 CSV
    os.makedirs("outputs", exist_ok=True)
    out_csv = args.out_csv

    # 遍历并运行
    for arch, dirs in [("resnet18", resnet_dirs), ("vgg_cifar", vgg_dirs)]:
        for d in dirs:
            if not os.path.isdir(d):
                print(f"跳过不存在的目录: {d}")
                continue

            print(f"\n 开始评估 {arch} | {d}")
            cmd = [
                "python", "scripts/eval.py",
                "--arch", arch,
                "--model-dir", d,
                "--out-csv", out_csv,
                "--batch-size", str(args.batch_size),
                "--pgd-steps", str(args.pgd_steps)
            ] + ["--pgd-eps-list"] + [str(eps) for eps in args.pgd_eps_list]

            # 调用 evaluate_models.py
            subprocess.run(cmd)

    print(f"\n 全部评估完成，结果保存在: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-csv', type=str, default="outputs/results.csv")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument('--pgd-eps-list', nargs='+', type=float, default=[2/255, 4/255, 8/255])
    args = parser.parse_args()
    main(args)
