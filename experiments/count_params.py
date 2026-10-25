# -*- coding: utf-8 -*-
"""统计各模型 best.pt 的总参数量 (用于消融表)"""
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

RUNS = {
    "baseline": "runs/train/baseline_yolov8n/weights/best.pt",
    "ca_shallow(2,4)": "runs/train/ca-shallow-real/weights/best.pt",
    "ca_deep(6,8)": "runs/train/ca-deep-real2/weights/best.pt",
    "ca_all(2,4,6,8)": "runs/train/ca-all-real/weights/best.pt",
    "ca_insert(neck)": "runs/train/train3/weights/best.pt",
}
lines = []
for name, rel in RUNS.items():
    p = ROOT / rel
    if not p.exists():
        lines.append(f"{name}: missing")
        continue
    m = YOLO(str(p))
    n = sum(x.numel() for x in m.model.parameters())
    lines.append(f"{name}: {n:,}")
res = "\n".join(lines)
(ROOT / "params_count.txt").write_text(res, encoding="utf-8")
print(res)
