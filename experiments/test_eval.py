# -*- coding: utf-8 -*-
"""
在官方测试集 (test, 19150 张) 上重新评估所有模型, 输出整体 + 分类别指标到 JSON。
结果保存到 runs/test/test_<name>, 不覆盖训练产物; 汇总写入 results/test_metrics.json。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordatt import register_custom_modules

register_custom_modules()

# 关闭 AMP 检查避免联网下载 yolo26n.pt
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
}


def main():
    summary = {}
    for name, rel in MODELS.items():
        wpath = ROOT / rel
        if not wpath.exists():
            print(f"[SKIP] {name}: {wpath} not found")
            continue
        print(f"\n===== evaluating {name} on test =====")
        model = YOLO(str(wpath))
        m = model.val(
            data=DATA,
            split="test",
            imgsz=640,
            batch=16,
            device="cuda:0",
            project=str(ROOT / "runs" / "test"),
            name=f"test_{name}",
            exist_ok=True,
            verbose=False,
        )
        ap50 = m.box.ap50.tolist()      # 分类别 AP50
        ap5095 = m.box.ap.tolist()      # 分类别 AP50-95 (8,10)
        entry = {
            "P": m.box.mp,
            "R": m.box.mr,
            "mAP50": m.box.map50,
            "mAP50-95": m.box.map,
            "speed_inference_ms": m.speed.get("inference"),
            "speed_total_ms": (m.speed.get("preprocess", 0) + m.speed.get("inference", 0)
                               + m.speed.get("postprocess", 0)),
            "class_ap50": {CLASS_EN[i]: round(float(ap50[i]), 4) for i in range(len(CLASS_EN))},
            "class_ap50_zh": {CLASS_ZH[i]: round(float(ap50[i]), 4) for i in range(len(CLASS_EN))},
            "class_ap5095": {CLASS_EN[i]: round(float(ap5095[i][-1]), 4) for i in range(len(CLASS_EN))},
        }
        summary[name] = entry
        print(f"{name}: mAP50={entry['mAP50']:.4f} mAP50-95={entry['mAP50-95']:.4f} "
              f"speed={entry['speed_inference_ms']:.2f}ms")

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsaved ->", outdir / "test_metrics.json")


if __name__ == "__main__":
    main()
