# -*- coding: utf-8 -*-
"""
在验证集 (val, 898 张) 上评估各模型, 输出整体 + 分类别 AP50 到 JSON。
用于论文图4-2 分类别对比与表4-1, 口径为 val (best.pt)。
与训练并行时用较小 batch。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordatt import register_custom_modules

register_custom_modules()

try:
    from ultralytics.utils import checks
    checks.check_amp = lambda model: True
except Exception:
    pass

from ultralytics import YOLO

DATA = str(ROOT / "config" / "pvelad.yaml")
CLASS_EN = ["finger", "crack", "black_core", "thick_line",
            "horizontal_dislocation", "short_circuit", "vertical_dislocation", "star_crack"]
CLASS_ZH = ["断栅", "裂纹", "黑芯", "增厚栅线", "横向错位", "短路", "纵向错位", "星形裂纹"]

MODELS = {
    "baseline": "runs/train/baseline_yolov8n/weights/best.pt",
    "ca_all_real": "runs/train/ca-all-real/weights/best.pt",
    "ca_deep_real2": "runs/train/ca-deep-real2/weights/best.pt",
    "ca_shallow_real": "runs/train/ca-shallow-real/weights/best.pt",
}


def main():
    summary = {}
    for name, rel in MODELS.items():
        wpath = ROOT / rel
        if not wpath.exists():
            print(f"[SKIP] {name}: not found")
            continue
        print(f"===== val {name} =====")
        model = YOLO(str(wpath))
        m = model.val(
            data=DATA,
            split="val",
            imgsz=640,
            batch=8,
            device="cuda:0",
            project=str(ROOT / "runs" / "val_eval"),
            name=f"val_{name}",
            exist_ok=True,
            verbose=False,
        )
        ap50 = m.box.ap50.tolist()
        ap = m.box.ap  # 该版本为每类 AP50-95 (1D)
        ap95 = [float(ap[i]) for i in range(len(CLASS_EN))]
        entry = {
            "P": m.box.mp,
            "R": m.box.mr,
            "mAP50": m.box.map50,
            "mAP50-95": m.box.map,
            "speed_inference_ms": m.speed.get("inference"),
            "class_ap50": {CLASS_EN[i]: round(float(ap50[i]), 4) for i in range(len(CLASS_EN))},
            "class_ap50_zh": {CLASS_ZH[i]: round(float(ap50[i]), 4) for i in range(len(CLASS_EN))},
            "class_ap5095": {CLASS_EN[i]: round(ap95[i], 4) for i in range(len(CLASS_EN))},
        }
        summary[name] = entry
        print(f"{name}: mAP50={entry['mAP50']:.4f} mAP50-95={entry['mAP50-95']:.4f}")

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved ->", outdir / "val_metrics.json")


if __name__ == "__main__":
    main()
