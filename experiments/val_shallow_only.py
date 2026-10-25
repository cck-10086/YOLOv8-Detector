# -*- coding: utf-8 -*-
"""仅独立评估浅层 CA (ca-shallow-real), 结果合并进 val_metrics.json, 其余模型保持不变。"""
import json
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
NAME = "ca_shallow_real"
WPT = ROOT / "runs/train/ca-shallow-real/weights/best.pt"


def main():
    out_json = ROOT / "results" / "val_metrics.json"
    summary = {}
    if out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            summary = json.load(f)

    print(f"===== val {NAME} =====")
    model = YOLO(str(WPT))
    m = model.val(
        data=DATA,
        split="val",
        imgsz=640,
        batch=8,
        device="cuda:0",
        project=str(ROOT / "runs" / "val_eval"),
        name=f"val_{NAME}",
        exist_ok=True,
        verbose=False,
    )
    ap50 = m.box.ap50.tolist()
    ap = m.box.ap
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
    summary[NAME] = entry
    print(f"{NAME}: P={entry['P']:.4f} R={entry['R']:.4f} mAP50={entry['mAP50']:.4f} mAP50-95={entry['mAP50-95']:.4f}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved ->", out_json)


if __name__ == "__main__":
    main()
