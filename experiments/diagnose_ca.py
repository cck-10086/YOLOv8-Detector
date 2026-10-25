# -*- coding: utf-8 -*-
"""诊断: 确认 CA 注入前后模型参数/结构是否正确"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"E:\study\计算机视觉综合实践\课程论文\YOLOv8-Detector")
sys.path.insert(0, str(PROJECT_ROOT))

_LINES = []
def print(*a):
    s = " ".join(str(x) for x in a)
    _LINES.append(s)
    builtins_print(s)
import builtins as _b
builtins_print = _b.print

from ultralytics import YOLO
from src.coordatt import register_custom_modules, inject_coord_attention, CoordAtt, C2f_CA

register_custom_modules()
w = r"E:\study\计算机视觉综合实践\课程论文\weights\yolov8n.pt"

# 1) 注入前 (baseline)
model = YOLO(w)
model.model.eval()
base_params = sum(p.numel() for p in model.model.parameters())
ca_before = sum(1 for _ in model.model.modules() if isinstance(_, CoordAtt))
print(f"[BEFORE] total params = {base_params}, CoordAtt modules = {ca_before}")
print("[BEFORE] layer6 type =", type(model.model.model[6]).__name__,
      "| layer8 type =", type(model.model.model[8]).__name__)

# 2) 注入深层 CA (6,8)
info = inject_coord_attention(model.model, stages=[6, 8])
print("[INJECT]", info)

deep_params = sum(p.numel() for p in model.model.parameters())
ca_after = [m for m in model.model.modules() if isinstance(m, CoordAtt)]
ca_params = sum(p.numel() for m in ca_after for p in m.parameters())
print(f"[AFTER] total params = {deep_params}, delta = {deep_params - base_params}")
print(f"[AFTER] CoordAtt modules = {len(ca_after)}, CA params = {ca_params}")
print("[AFTER] layer6 type =", type(model.model.model[6]).__name__,
      "| layer8 type =", type(model.model.model[8]).__name__)
print("[AFTER] layer6.ca type =", type(model.model.model[6].ca).__name__,
      "| layer8.ca type =", type(model.model.model[8].ca).__name__)

# 3) 前向测试 (确认无报错)
import torch
x = torch.randn(2, 3, 640, 640, device="cuda:0")
model.model.to("cuda:0")
model.model.train()
with torch.no_grad():
    out = model.model(x)
print("[FORWARD] ok, output heads =", len(out) if isinstance(out, (list, tuple)) else type(out))

out_file = PROJECT_ROOT / "diagnose_result.txt"
out_file.write_text("\n".join(_LINES), encoding="utf-8")
builtins_print("written to", out_file)
