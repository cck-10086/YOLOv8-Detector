# -*- coding: utf-8 -*-
"""提取所有训练 run 的关键指标: 末轮 + 峰值 mAP50 及对应轮次"""
import csv
from pathlib import Path

ROOT = Path(r"E:\study\计算机视觉综合实践\课程论文\YOLOv8-Detector")
RUNS = ROOT / "runs" / "train"
RUN_NAMES = ["baseline_yolov8n", "train3", "ca-all-real", "ca-deep-real2", "ca-shallow-real"]

out = []
for name in RUN_NAMES:
    csvp = RUNS / name / "results.csv"
    if not csvp.exists():
        out.append(f"## {name}: no results.csv")
        continue
    rows = []
    with open(csvp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    if not rows:
        out.append(f"## {name}: empty")
        continue
    last = rows[-1]
    # 峰值 mAP50
    peak = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    # 峰值 mAP50-95
    peak95 = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    out.append(f"## {name}  (epochs={len(rows)})")
    out.append("  last  : ep{:>3} P={:.4f} R={:.4f} mAP50={:.4f} mAP50-95={:.4f}".format(
        last["epoch"], float(last["metrics/precision(B)"]), float(last["metrics/recall(B)"]),
        float(last["metrics/mAP50(B)"]), float(last["metrics/mAP50-95(B)"])))
    out.append("  peak50: ep{:>3} mAP50={:.4f} (P={:.4f} R={:.4f} mAP50-95={:.4f})".format(
        peak["epoch"], float(peak["metrics/mAP50(B)"]), float(peak["metrics/precision(B)"]),
        float(peak["metrics/recall(B)"]), float(peak["metrics/mAP50-95(B)"])))
    out.append("  peak95: ep{:>3} mAP50-95={:.4f} (mAP50={:.4f})".format(
        peak95["epoch"], float(peak95["metrics/mAP50-95(B)"]), float(peak95["metrics/mAP50(B)"])))

text = "\n".join(out)
(ROOT / "metrics_summary.txt").write_text(text, encoding="utf-8")
print(text)
