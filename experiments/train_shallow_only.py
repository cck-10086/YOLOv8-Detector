# -*- coding: utf-8 -*-
"""单独训练浅层 CA (ca-shallow-real), 全新 100 轮。"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(r"E:\study\计算机视觉综合实践\课程论文\YOLOv8-Detector")
sys.path.insert(0, str(PROJECT_ROOT))

W = r"E:\study\计算机视觉综合实践\课程论文\weights\yolov8n.pt"
DATA = str(PROJECT_ROOT / "config" / "pvelad.yaml")
STAGES = [2, 4]
NAME = "ca-shallow-real"


def main():
    import ultralytics.utils.checks as checks
    checks.check_amp = lambda model: True

    from src.coordatt import register_custom_modules, inject_coord_attention
    register_custom_modules()

    from ultralytics.models.yolo.detect import DetectionTrainer
    orig_get_model = DetectionTrainer.get_model

    def patched_get_model(self, cfg=None, weights=None, verbose=True):
        m = orig_get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        info = inject_coord_attention(m, stages=STAGES)
        print("[INJECT]", info, flush=True)
        return m

    DetectionTrainer.get_model = patched_get_model

    from ultralytics import YOLO
    model = YOLO(W)
    model.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=16,
        device="cuda:0",
        workers=4,
        patience=25,
        seed=0,
        amp=True,
        name=NAME,
        exist_ok=True,
        project=str(PROJECT_ROOT / "runs" / "train"),
    )
    print("[DONE]", NAME, flush=True)


if __name__ == "__main__":
    main()
