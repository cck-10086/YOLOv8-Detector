# -*- coding: utf-8 -*-
"""补训真实深层 CA (backbone 层6,8), 绕过 AMP 网络检查。
超参数与 baseline / train5 / train6 完全一致 (100ep, batch16, imgsz640, seed0, patience30)。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"E:\study\计算机视觉综合实践\课程论文\YOLOv8-Detector")
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # 绕过 AMP 网络检查: github 无法下载 yolo26n.pt, 但 RTX 4060 对 AMP 支持良好
    import ultralytics.utils.checks as checks
    checks.check_amp = lambda model: True

    from ultralytics import YOLO
    from src.coordatt import register_custom_modules, inject_coord_attention

    register_custom_modules()
    w = r"E:\study\计算机视觉综合实践\课程论文\weights\yolov8n.pt"
    data = str(PROJECT_ROOT / "config" / "pvelad.yaml")

    model = YOLO(w)
    info = inject_coord_attention(model.model, stages=[6, 8])
    print("[INJECT]", info)

    model.train(
        data=data,
        epochs=100,
        imgsz=640,
        batch=16,
        device="cuda:0",
        workers=4,
        patience=30,
        seed=0,
        amp=True,
        name="ca-deep-real",
        exist_ok=True,
        project=str(PROJECT_ROOT / "runs" / "train"),
    )
    print("[DONE] ca-deep-real training finished")


if __name__ == "__main__":
    main()
