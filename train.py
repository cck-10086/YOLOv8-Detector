#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 光伏电池EL缺陷检测 - 训练入口

功能:
  1. 基线模型训练 (YOLOv8n/s, COCO 预训练权重迁移学习)
  2. 改进模型训练 (嵌入坐标注意力的 yolov8n-ca.yaml)
  3. 训练完成后自动在验证集上评估并输出指标

用法:
  # 训练基线模型
  python train.py --model yolov8n

  # 训练改进模型 (坐标注意力)
  python train.py --model config/yolov8n-ca.yaml

  # 自定义超参数
  python train.py --model yolov8n --epochs 100 --batch 16 --imgsz 640

  # 从断点恢复训练
  python train.py --resume runs/detect/train/weights/last.pt
"""

import os
import sys
import argparse

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 光伏EL缺陷检测训练系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py --model yolov8n
  python train.py --model config/yolov8n-ca.yaml
  python train.py --model yolov8s --epochs 100 --batch 16
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help="模型名称或结构配置文件路径: yolov8n, yolov8s, config/yolov8n-ca.yaml (默认: yolov8n)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="config/pvelad.yaml",
        help="数据集配置文件 (默认: config/pvelad.yaml)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数 (默认: 100)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸 (默认: 640)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批大小 (默认: 16, 8GB显存建议不超过16)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="训练设备 (默认: cuda:0, 可选 cpu)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="数据加载线程数 (默认: 4, Windows下不建议过大)",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="本次训练运行名称 (默认: 自动按模型名+时间生成)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="早停轮数 (默认: 30)",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="自定义结构(.yaml)时加载的预训练权重 (默认: yolov8n.pt, 传 none 从头训练)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从断点恢复训练 (传入 last.pt 路径)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"  YOLOv8 光伏EL缺陷检测 - 训练")
    print(f"  模型: {args.model} | 数据: {args.data}")
    print(f"  轮数: {args.epochs} | 批大小: {args.batch} | 尺寸: {args.imgsz}")
    print(f"  设备: {args.device}")
    print(f"{'#'*60}\n")

    from ultralytics import YOLO

    # 若使用改进结构, 注册自定义模块 (CoordAtt), 使 YAML 可引用
    if "ca.yaml" in args.model.lower():
        from src.coordatt import register_custom_modules
        register_custom_modules()

    # 路径处理: 相对路径以项目根目录为基准解析
    model_arg = args.model if os.path.isabs(args.model) else os.path.join(PROJECT_ROOT, args.model)
    data_arg = args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data)

    # 构建模型: .pt 为预训练权重迁移学习, .yaml 为从头构建结构
    model = YOLO(args.resume) if args.resume else YOLO(model_arg)

    # 自定义结构加载匹配的预训练权重 (backbone 与基线一致, 保证与基线对比公平)
    if not args.resume and str(model_arg).lower().endswith(".yaml"):
        pretrained = args.pretrained if args.pretrained else "yolov8n.pt"
        if pretrained.lower() != "none":
            model.load(pretrained)
            print(f"[INFO] 已加载预训练权重: {pretrained} (匹配层迁移)")

    # 启动训练 (训练完成后 Ultralytics 自动在 val 集评估)
    # project 使用绝对路径, 避免 Ultralytics 将相对路径解析到其全局 runs_dir
    model.train(
        data=data_arg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=args.patience,
        seed=0,
        project=os.path.join(PROJECT_ROOT, "runs", "train"),
    )

    print(f"\n[完成] 训练结束! 结果保存在: {os.path.join(PROJECT_ROOT, 'runs', 'train')}")


if __name__ == "__main__":
    main()
