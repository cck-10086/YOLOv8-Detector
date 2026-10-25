# -*- coding: utf-8 -*-
"""诊断每个模型的 CA 模块数量、类型分布与总参数"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.coordatt import register_custom_modules, CoordAtt

register_custom_modules()

try:
    from ultralytics.utils import checks
    checks.check_amp = lambda model: True
except Exception:
    pass

from ultralytics import YOLO
import torch

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
        lines.append(f"## {name}: missing")
        continue
    try:
        m = YOLO(str(p))
        total = sum(x.numel() for x in m.model.parameters())
        ca_modules = [x for x in m.model.modules() if isinstance(x, CoordAtt)]
        ca_params = sum(x.numel() for x in ca_modules)
        type_counts = {}
        for x in m.model.modules():
            tn = type(x).__name__
            type_counts[tn] = type_counts.get(tn, 0) + 1
        lines.append(f"## {name}: total={total:,} CA_count={len(ca_modules)} CA_params={ca_params}")
        lines.append("   types: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())
                                              if k in ("C2f", "C2f_CA", "CoordAtt")))
    except Exception as e:
        import traceback
        lines.append(f"## {name}: ERROR {e}")
        lines.append(traceback.format_exc()[:600])
res = "\n".join(lines)
(ROOT / "module_diag.txt").write_text(res, encoding="utf-8")
print(res)
