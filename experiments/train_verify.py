# -*- coding: utf-8 -*-
"""快速验证修复: 训练2轮, 检查 best.pt 是否真正含 C2f_CA (全部逻辑置于 main)"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"E:\study\计算机视觉综合实践\课程论文\YOLOv8-Detector")
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import ultralytics.utils.checks as checks
    checks.check_amp = lambda model: True

    from src.coordatt import register_custom_modules, inject_coord_attention
    register_custom_modules()

    from ultralytics.models.yolo.detect import DetectionTrainer
    orig = DetectionTrainer.get_model

    def patched(self, cfg=None, weights=None, verbose=True):
        m = orig(self, cfg=cfg, weights=weights, verbose=verbose)
        print("[INJECT]", inject_coord_attention(m, stages=[2, 4, 6, 8]), flush=True)
        return m

    DetectionTrainer.get_model = patched

    from ultralytics import YOLO
    model = YOLO(r"E:\study\计算机视觉综合实践\课程论文\weights\yolov8n.pt")
    model.train(
        data=str(PROJECT_ROOT / "config" / "pvelad.yaml"),
        epochs=2,
        imgsz=640,
        batch=16,
        device="cuda:0",
        workers=4,
        patience=25,
        seed=0,
        name="verify-ca",
        exist_ok=True,
        project=str(PROJECT_ROOT / "runs" / "train"),
    )
    print("[VERIFY DONE]", flush=True)


if __name__ == "__main__":
    main()
