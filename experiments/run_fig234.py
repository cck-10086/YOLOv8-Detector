# -*- coding: utf-8 -*-
"""临时: 仅生成 fig2 / fig3 / fig4 (fig1 等全部训练完成后再生成)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import visualize

print("[fig2]", visualize.fig2_class_ap())
print("[fig3]", visualize.fig3_class_distribution())
samples = visualize.pick_samples(3)
print("[fig4]", visualize.fig4_detection_compare(samples))
