# -*- coding: utf-8 -*-
"""
正确训练真正含 CA 的模型 (全层/深层/浅层), 顺序执行。

修复点: monkey-patch DetectionTrainer.get_model, 使 Trainer 在构建模型阶段
(setup_model -> get_model) 就注入 CA, 保证训练、保存的 best.pt 真正含 C2f_CA。
超参数与 baseline 一致 (100ep, batch16, imgsz640, seed0, patience25)。
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(r"E:\study\计算机视觉综合实践\课程论文\YOLOv8-Detector")
sys.path.insert(0, str(PROJECT_ROOT))

W = r"E:\study\计算机视觉综合实践\课程论文\weights\yolov8n.pt"
DATA = str(PROJECT_ROOT / "config" / "pvelad.yaml")

JOBS = [
    ("ca-all-real", [2, 4, 6, 8]),
    ("ca-deep-real2", [6, 8]),
    ("ca-shallow-real", [2, 4]),
]


def main():
    # 绕过 AMP 检查 (github 无法下载 yolo26n.pt)
    import ultralytics.utils.checks as checks
    checks.check_amp = lambda model: True

    from src.coordatt import register_custom_modules, inject_coord_attention
    register_custom_modules()

    # monkey-patch get_model
    from ultralytics.models.yolo.detect import DetectionTrainer
    orig_get_model = DetectionTrainer.get_model
    state = {"stages": None}

    def patched_get_model(self, cfg=None, weights=None, verbose=True):
        m = orig_get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        if state["stages"]:
            info = inject_coord_attention(m, stages=state["stages"])
            print("[INJECT]", info, flush=True)
        return m

    DetectionTrainer.get_model = patched_get_model

    from ultralytics import YOLO
    for name, stages in JOBS:
        print(f"\n========== {name} stages={stages} ==========", flush=True)
        state["stages"] = stages
        try:
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
                name=name,
                exist_ok=True,
                project=str(PROJECT_ROOT / "runs" / "train"),
            )
            print(f"[DONE] {name}", flush=True)
        except Exception as e:
            print(f"[ERROR] {name}: {e}", flush=True)
            traceback.print_exc()
            state["stages"] = None
    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
