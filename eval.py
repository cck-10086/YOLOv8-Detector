#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 光伏电池EL缺陷检测 - 评估入口

功能:
  1. 在验证集/测试集上评估训练好的模型
  2. 输出精确率、召回率、mAP50、mAP50-95 等指标
  3. 生成混淆矩阵、PR曲线等可视化结果 (保存在 runs/val 目录)

用法:
  # 在验证集上评估
  python eval.py --model runs/train/yolov8n/weights/best.pt

  # 在测试集上评估
  python eval.py --model runs/train/yolov8n/weights/best.pt --split test

  # 对比基线与改进模型 (自动生成对比表格)
  python eval.py --model runs/train/yolov8n/weights/best.pt runs/train/yolov8n-ca/weights/best.pt --split test
"""

import os
import sys
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 光伏EL缺陷检测评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eval.py --model runs/train/yolov8n/weights/best.pt
  python eval.py --model runs/train/yolov8n-ca/weights/best.pt --split test
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="模型权重路径 (.pt), 可传入多个以进行对比评估",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="config/pvelad.yaml",
        help="数据集配置文件 (默认: config/pvelad.yaml)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="评估的数据集划分 (默认: val, 可选 test)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸 (默认: 640, 需与训练一致)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批大小 (默认: 16)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="评估设备 (默认: cuda:0, 可选 cpu)",
    )

    return parser.parse_args()


def main():
    """主函数: 逐个评估模型并输出对比表格"""
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"  YOLOv8 光伏EL缺陷检测 - 评估")
    print(f"  模型数: {len(args.model)} | 划分: {args.split} | 设备: {args.device}")
    print(f"{'#'*60}\n")

    from ultralytics import YOLO

    # 数据配置路径以脚本所在目录 (项目根) 为基准解析
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_arg = args.data if os.path.isabs(args.data) else os.path.join(project_root, args.data)

    results_list = []

    for i, model_path in enumerate(args.model, 1):
        if not os.path.exists(model_path):
            print(f"[ERROR] 模型文件不存在: {model_path}")
            sys.exit(1)

        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        print(f"\n[{i}/{len(args.model)}] 正在评估: {model_name} ({model_path})")

        model = YOLO(model_path)
        metrics = model.val(
            data=data_arg,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=f"val_{model_name}",
            project=os.path.join(project_root, "runs", "val"),  # 绝对路径避免解析到全局 runs_dir
        )

        results_list.append({
            "name": model_name,
            "P": metrics.box.mp,          # 平均精确率
            "R": metrics.box.mr,          # 平均召回率
            "mAP50": metrics.box.map50,   # mAP@0.5
            "mAP50-95": metrics.box.map,  # mAP@0.5:0.95
            "speed": metrics.speed["inference"],  # 推理速度 ms/张
        })

    # 打印对比表格
    print(f"\n{'='*72}")
    print(f"  评估结果对比 ({args.split} 集)")
    print(f"{'='*72}")
    print(f"  {'模型':<24} {'P':<8} {'R':<8} {'mAP50':<8} {'mAP50-95':<10} {'速度(ms)':<8}")
    print(f"  {'-'*72}")
    for r in results_list:
        print(
            f"  {r['name']:<24} {r['P']:<8.4f} {r['R']:<8.4f} "
            f"{r['mAP50']:<8.4f} {r['mAP50-95']:<10.4f} {r['speed']:<8.1f}"
        )
    print(f"{'='*72}")
    print(f"[提示] 混淆矩阵/PR曲线等可视化结果保存在: runs/val/")


if __name__ == "__main__":
    main()
