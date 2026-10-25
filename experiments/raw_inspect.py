# -*- coding: utf-8 -*-
"""绕过 YOLO 封装, 直接 torch.load best.pt, 检查真实保存的层类型"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.coordatt import register_custom_modules

register_custom_modules()
import torch

FILES = {
    "ca-all-real": "runs/train/ca-all-real/weights/best.pt",
}
lines = []
for name, rel in FILES.items():
    p = ROOT / rel
    try:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        lines.append(f"## {name}: LOAD ERROR {e}")
        continue
    model = ckpt["model"]
    lines.append(f"## {name}")
    layers = model.model if hasattr(model, "model") else model
    for i, layer in enumerate(layers):
        tn = type(layer).__name__
        mark = "  <<< CA" if ("CA" in tn or "Coord" in tn) else ""
        lines.append(f"   layer{i:>2}: {tn}{mark}")
    ca = [type(x).__name__ for x in model.modules() if "Coord" in type(x).__name__ or "C2f_CA" in type(x).__name__]
    lines.append(f"   CA-like modules: {ca}")
res = "\n".join(lines)
(ROOT / "raw_inspect_result.txt").write_text(res, encoding="utf-8")
print(res)
