# -*- coding: utf-8 -*-
"""
ONNX 导出与精度一致性验证 (部署第一步)

流程:
  1. 加载最优 PyTorch 模型 -> 导出 ONNX (CPU 执行, 不干扰后台训练)
  2. 对比 .pt 与 .onnx 文件大小
  3. 用 onnxruntime 后端在验证集上完整评估, 与 PyTorch 后端指标对比,
     验证导出后精度无损
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.coordatt import register_custom_modules  # noqa: E402

# best.pt 中 pickle 了 C2f_CA 自定义类, 加载前必须注册
register_custom_modules()

from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 并验证精度一致性")
    parser.add_argument("--model", type=str,
                        default=os.path.join(PROJECT_ROOT, "runs/train/yolov8n-ca22/weights/best.pt"),
                        help="待导出的 PyTorch 权重路径")
    parser.add_argument("--data", type=str,
                        default=os.path.join(PROJECT_ROOT, "config/pvelad.yaml"),
                        help="数据集配置 (验证用)")
    args = parser.parse_args()

    # ---------- 1. 导出 ----------
    model = YOLO(args.model)
    onnx_path = model.export(format="onnx", imgsz=640, opset=12, device="cpu",
                             dynamic=False, simplify=False)
    print(f"\n[导出完成] {onnx_path}")

    # ---------- 2. 文件大小对比 ----------
    pt_size = os.path.getsize(args.model) / 1e6
    onnx_size = os.path.getsize(onnx_path) / 1e6
    print(f"文件大小: .pt {pt_size:.1f}MB -> .onnx {onnx_size:.1f}MB "
          f"({onnx_size / pt_size * 100:.1f}%)")

    # ---------- 3. 精度一致性 (验证集, 双后端同配置 CPU 对比) ----------
    # 两侧均用 batch=1 (方形 letterbox), 保证唯一差异是推理后端
    print("\n[PyTorch 后端验证] 评估中 (CPU, batch=1)...")
    pt_metrics = YOLO(args.model).val(
        data=args.data, split="val", device="cpu", imgsz=640, batch=1, verbose=False)

    print("[ONNX 后端验证] 评估中 (onnxruntime CPU)...")
    onnx_model = YOLO(onnx_path, task="detect")
    onnx_metrics = onnx_model.val(
        data=args.data, split="val", device="cpu", imgsz=640, batch=1, verbose=False)

    print("\n==================== 精度一致性对比 (验证集 898 张) ====================")
    print(f"{'后端':<16}{'P':<10}{'R':<10}{'mAP50':<10}{'mAP50-95':<10}")
    print(f"{'PyTorch':<16}{pt_metrics.box.mp:<10.3f}{pt_metrics.box.mr:<10.3f}"
          f"{pt_metrics.box.map50:<10.3f}{pt_metrics.box.map:<10.3f}")
    print(f"{'onnxruntime':<16}{onnx_metrics.box.mp:<10.3f}{onnx_metrics.box.mr:<10.3f}"
          f"{onnx_metrics.box.map50:<10.3f}{onnx_metrics.box.map:<10.3f}")
    delta = abs(onnx_metrics.box.map50 - pt_metrics.box.map50)
    print(f"\nmAP50 绝对偏差: {delta:.4f} ({'一致, 精度无损' if delta < 0.005 else '存在偏差, 需排查'})")


if __name__ == "__main__":
    main()
