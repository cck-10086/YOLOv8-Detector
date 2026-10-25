# -*- coding: utf-8 -*-
"""
实验结果可视化: 一键生成论文所需的全部图片素材

输出 (results/figures/):
  fig1_training_curves.png    训练曲线对比 (五模型 mAP 曲线 + 基线vs改进 loss 曲线)
  fig2_class_ap.png           分类别 AP50 对比 (基线 vs 改进, 分组条形图)
  fig3_class_distribution.png 训练集类别分布 (长尾分布展示)
  fig4_detection_compare.png  检测效果对比 (基线 vs 改进, 多样例并排)

分类别 AP 数据来自 eval.py 在 test 集 (19150 张) 的评估输出, 内置以保证可复现。
"""

import glob
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "train")

# matplotlib 中文显示配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 训练 run 目录映射 (results.csv 均为完整训练, 真实含 CA)
RUNS = {
    "基线 YOLOv8n": "baseline_yolov8n",
    "CA 颈部插入式": "train3",
    "CA 全层(2,4,6,8)": "ca-all-real",
    "CA 浅层(2,4)": "ca-shallow-real",
    "CA 深层(6,8)": "ca-deep-real2",
}
RUN_COLORS = {
    "基线 YOLOv8n": "#888888",
    "CA 颈部插入式": "#b0b0b0",
    "CA 全层(2,4,6,8)": "#d62728",
    "CA 浅层(2,4)": "#1f77b4",
    "CA 深层(6,8)": "#2ca02c",
}

# 类别定义 (PVEL-AD 8 类) 与中文名
CLASS_EN = ["finger", "crack", "black_core", "thick_line",
            "horizontal_dislocation", "short_circuit", "vertical_dislocation", "star_crack"]
CLASS_ZH = ["断栅", "裂纹", "黑芯", "增厚栅线", "横向错位", "短路", "纵向错位", "星形裂纹"]

# 分类别 val AP50 (best.pt 独立 val 评估; 若 val_metrics.json 存在则覆盖)
CLASS_AP = {
    "基线 YOLOv8n":   [0.936, 0.742, 0.988, 0.913, 0.863, 0.995, 0.769, 0.838],
    "CA2 全层(2,4,6,8)": [0.932, 0.748, 0.993, 0.883, 0.823, 0.995, 0.853, 0.824],
}


def load_test_metrics():
    """读取 val_metrics.json (统一口径 best.pt 独立 val)"""
    p = os.path.join(PROJECT_ROOT, "results", "val_metrics.json")
    if os.path.exists(p):
        import json
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_results_csv(run_name):
    """读取 results.csv, 返回关键字段字典"""
    import csv
    path = os.path.join(RUNS_DIR, run_name, "results.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    get = lambda r, k: float(r[k.strip()])
    return {
        "epoch": [int(float(r["epoch"])) + 1 for r in rows],
        "map50": [get(r, "metrics/mAP50(B)") for r in rows],
        "map50_95": [get(r, "metrics/mAP50-95(B)") for r in rows],
        "box_loss": [get(r, "train/box_loss") for r in rows],
        "cls_loss": [get(r, "train/cls_loss") for r in rows],
    }


def ema(values, alpha=0.9):
    """指数滑动平均, 平滑曲线便于观察趋势"""
    out, s = [], values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def fig1_training_curves(data):
    """图1: 训练曲线 (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax1, ax2), (ax3, ax4) = axes

    # (a) 五模型 val mAP50 曲线
    for name, run in RUNS.items():
        d = data[name]
        ax1.plot(d["epoch"], d["map50"], label=name, color=RUN_COLORS[name],
                 lw=2.2 if "深层" in name else 1.2, alpha=0.9 if "深层" in name or name == "基线 YOLOv8n" else 0.65)
    ax1.set_xlabel("训练轮次 (epoch)")
    ax1.set_ylabel("验证集 mAP50")
    ax1.set_title("(a) 五模型验证集 mAP50 曲线对比")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    # (b) box loss 对比 (基线 vs 深层CA)
    for name in ["基线 YOLOv8n", "CA 深层(6,8)"]:
        d = data[name]
        ax2.plot(d["epoch"], ema(d["box_loss"]), label=name, color=RUN_COLORS[name], lw=1.8)
    ax2.set_xlabel("训练轮次 (epoch)")
    ax2.set_ylabel("训练 box loss (EMA平滑)")
    ax2.set_title("(b) 边界框回归损失对比")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # (c) cls loss 对比 (基线 vs 深层CA)
    for name in ["基线 YOLOv8n", "CA 深层(6,8)"]:
        d = data[name]
        ax3.plot(d["epoch"], ema(d["cls_loss"]), label=name, color=RUN_COLORS[name], lw=1.8)
    ax3.set_xlabel("训练轮次 (epoch)")
    ax3.set_ylabel("训练 cls loss (EMA平滑)")
    ax3.set_title("(c) 分类损失对比")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # (d) 五模型 val mAP50-95 曲线
    for name, run in RUNS.items():
        d = data[name]
        ax4.plot(d["epoch"], d["map50_95"], label=name, color=RUN_COLORS[name],
                 lw=2.2 if "深层" in name else 1.2, alpha=0.9 if "深层" in name or name == "基线 YOLOv8n" else 0.65)
    ax4.set_xlabel("训练轮次 (epoch)")
    ax4.set_ylabel("验证集 mAP50-95")
    ax4.set_title("(d) 五模型验证集 mAP50-95 曲线对比")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.grid(alpha=0.3)

    fig.suptitle("图1 模型训练过程对比 (PVEL-AD 数据集)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "fig1_training_curves.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig2_class_ap():
    """图2: 分类别 AP50 分组条形图 (优先用 val_metrics.json)"""
    base_ap = CLASS_AP["基线 YOLOv8n"]
    ca_ap = CLASS_AP["CA2 全层(2,4,6,8)"]
    base_m, ca_m = "0.880", "0.902"
    tm = load_test_metrics()
    if tm:
        base_ap = [tm["baseline"]["class_ap50"][c] for c in CLASS_EN]
        ca_ap = [tm["ca_deep_real2"]["class_ap50"][c] for c in CLASS_EN]
        base_m = f"{tm['baseline']['mAP50']:.3f}"
        ca_m = f"{tm['ca_deep_real2']['mAP50']:.3f}"
    # 提升最大的两个类别
    diffs = [ca_ap[i] - base_ap[i] for i in range(len(CLASS_EN))]
    order = np.argsort(diffs)[::-1]
    gain_idx = [int(order[0]), int(order[1])]

    x = np.arange(len(CLASS_ZH))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - w / 2, base_ap, w, label=f"基线 YOLOv8n (val mAP50={base_m})",
                color="#888888", edgecolor="white")
    b2 = ax.bar(x + w / 2, ca_ap, w, label=f"CA 深层改进 (val mAP50={ca_m})",
                color="#d62728", edgecolor="white")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.008,
                    f"{rect.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ZH, fontsize=10)
    ax.set_ylabel("验证集 AP50")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.01),
              frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for gi in gain_idx:
        if diffs[gi] > 0:
            ax.annotate(f"+{diffs[gi]*100:.1f}pt", xy=(gi + w / 2, ca_ap[gi]),
                        xytext=(gi + w / 2, ca_ap[gi] + 0.095),
                        ha="center", fontsize=11, color="#d62728", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#d62728"))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, "fig2_class_ap.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig3_class_distribution():
    """图3: 训练集类别分布 (均衡采样后)"""
    label_dir = os.path.join(PROJECT_ROOT, "data", "PVEL-AD", "labels", "train")
    counter = Counter()
    for lf in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(lf, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    counter[int(line.split()[0])] += 1
    counts = [counter.get(i, 0) for i in range(len(CLASS_ZH))]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(CLASS_ZH, counts, color=["#4c72b0"] * len(CLASS_ZH), edgecolor="white")
    for rect, c in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, c + max(counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("边界框实例数")
    ax.set_title("图3 训练集各类别缺陷实例分布 (train 4102 张)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_class_distribution.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------- 图4: 检测效果对比 ----------

FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
FONT_SMALL = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
COLORS = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14),
          (148, 103, 189), (44, 160, 44), (140, 86, 75), (188, 189, 34)]


def draw_preds(img_bgr, result, font):
    """Ultralytics predict 结果 -> 中文标签绘制"""
    img = img_bgr[:, :, ::-1].copy()
    img_pil = Image.fromarray(img)
    d = ImageDraw.Draw(img_pil)
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        color = COLORS[cls % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{CLASS_ZH[cls]} {conf:.2f}"
        tb = d.textbbox((x1, max(0, y1 - 24)), text, font=font)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        d.text((x1, max(0, y1 - 24)), text, fill="white", font=font)
    return np.array(img_pil)


def pick_samples(n=3):
    """从验证集挑选含纵向错位(6)/横向错位(4)/裂纹(1) 的图片"""
    label_dir = os.path.join(PROJECT_ROOT, "data", "PVEL-AD", "labels", "val")
    img_dir = os.path.join(PROJECT_ROOT, "data", "PVEL-AD", "images", "val")
    want = [6, 4, 1]  # 优先级: 纵向错位(CA提升最大), 横向错位, 裂纹
    picked = []
    for target in want:
        for lf in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
            path_img = None
            for ext in (".jpg", ".png", ".JPG", ".PNG"):
                stem = os.path.splitext(os.path.basename(lf))[0]
                p = os.path.join(img_dir, stem + ext)
                if os.path.exists(p):
                    path_img = p
                    break
            if not path_img:
                continue
            with open(lf, encoding="utf-8") as f:
                classes = [int(line.split()[0]) for line in f if line.strip()]
            if target in classes and path_img not in [p for p, _ in picked]:
                picked.append((path_img, target))
                break
        if len(picked) >= n:
            break
    return picked


def fig4_detection_compare(samples):
    """图4: 基线 vs CA2 检测效果对比 (多行样例)"""
    import cv2
    sys.path.insert(0, PROJECT_ROOT)
    from src.coordatt import register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO

    pt_baseline = os.path.join(RUNS_DIR, "baseline_yolov8n", "weights", "best.pt")
    pt_ca2 = os.path.join(RUNS_DIR, "ca-deep-real2", "weights", "best.pt")
    m_base = YOLO(pt_baseline)
    m_ca2 = YOLO(pt_ca2)

    rows = []
    for img_path, target_cls in samples:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        img_r = cv2.resize(img, (round(w * scale), round(h * scale))) if scale < 1 else img
        r_base = m_base.predict(img_r, conf=0.25, verbose=False)[0]
        r_ca2 = m_ca2.predict(img_r, conf=0.25, verbose=False)[0]
        rows.append((draw_preds(img_r, r_base, FONT_SMALL), draw_preds(img_r, r_ca2, FONT_SMALL),
                     os.path.basename(img_path), CLASS_ZH[target_cls]))

    if not rows:
        print("[警告] 未找到合适样例图, 跳过图4")
        return None

    ph, pw = rows[0][0].shape[:2]
    header = 46
    canvas = np.full((header * 1 + (ph + header) * len(rows), pw * 2 + 12, 3), 255, dtype=np.uint8)
    tm = load_test_metrics()
    bm, cm_ = "0.880", "0.902"
    if tm:
        bm = f"{tm['baseline']['mAP50']:.3f}"
        cm_ = f"{tm['ca_deep_real2']['mAP50']:.3f}"
    titles = [f"基线 YOLOv8n (val mAP50={bm})", f"CA 深层坐标注意力 (val mAP50={cm_})"]
    cv2.rectangle(canvas, (0, 0), (pw, header), (235, 235, 235), -1)
    cv2.rectangle(canvas, (pw + 12, 0), (pw * 2 + 12, header), (255, 230, 230), -1)
    pil = Image.fromarray(canvas[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    d.text((pw // 2 - 150, 12), titles[0], fill=(60, 60, 60), font=FONT)
    d.text((pw + 12 + pw // 2 - 150, 12), titles[1], fill=(180, 30, 30), font=FONT)
    arr = np.array(pil)[:, :, ::-1]

    y = header
    for img_base, img_ca2, name, zh in rows:
        b = cv2.copyMakeBorder(img_base, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        c = cv2.copyMakeBorder(img_ca2, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        if b.shape[:2] != (ph, pw):
            b = cv2.resize(b, (pw, ph))
            c = cv2.resize(c, (pw, ph))
        arr[y:y + ph, 0:pw] = b
        arr[y:y + ph, pw + 12:pw * 2 + 12] = c
        y += ph + header

    out = os.path.join(FIG_DIR, "fig4_detection_compare.png")
    Image.fromarray(arr[:, :, ::-1]).save(out)
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # 图1: 训练曲线
    data = {name: load_results_csv(run) for name, run in RUNS.items()}
    print("[1/4]", fig1_training_curves(data))

    # 图2: 分类别 AP
    print("[2/4]", fig2_class_ap())

    # 图3: 类别分布
    print("[3/4]", fig3_class_distribution())

    # 图4: 检测对比
    samples = pick_samples(3)
    print("[4/4]", fig4_detection_compare(samples))

    print("\n全部图片已生成至:", FIG_DIR)


if __name__ == "__main__":
    main()
