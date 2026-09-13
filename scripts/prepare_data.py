#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PVEL-AD 数据集准备脚本

功能:
  1. 解压原始数据集 zip (可选)
  2. 解析 VOC XML 标注并转换为 YOLO txt 格式
  3. 统计类别分布 (长尾分布报告)
  4. 类别均衡采样: 包含稀缺类的图像全部保留, 高频类图像按上限降采样
  5. 从 trainval 中分层划分 train/val, test 沿用官方划分
  6. 生成 Ultralytics 数据配置 (自动更新 config/pvelad.yaml 的类别数)

用法:
  # 从 zip 开始 (自动解压)
  python scripts/prepare_data.py --zip data/raw/PVEL-AD.zip

  # 指定已解压的数据目录
  python scripts/prepare_data.py --src data/raw/PVEL-AD

  # 使用全部12类 (默认仅8类主要缺陷)
  python scripts/prepare_data.py --src data/raw/PVEL-AD --all-classes
"""

import os
import sys
import csv
import random
import shutil
import zipfile
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 类别定义: 8类主要缺陷 (与计划书一致) + 4类极稀缺类别
MAIN_CLASSES = [
    "finger", "crack", "black_core", "thick_line",
    "horizontal_dislocation", "short_circuit", "vertical_dislocation", "star_crack",
]
RARE_CLASSES = ["printing_error", "corner", "fragment", "scratch"]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PVEL-AD 数据集准备 (XML->YOLO转换 + 类别均衡采样 + 数据集划分)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--zip", type=str, default=None, help="原始数据集 zip/rar 路径 (自动解压)")
    parser.add_argument("--src", type=str, default=None, help="已解压的数据集目录 (EL2021 层级)")
    parser.add_argument("--out", type=str, default="data/PVEL-AD", help="输出目录 (默认: data/PVEL-AD)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例 (默认: 0.2)")
    parser.add_argument("--max-per-class", type=int, default=2000, help="高频类最大图像数 (默认: 2000)")
    parser.add_argument("--rare-threshold", type=int, default=500,
                        help="稀缺类实例数阈值, 低于该值的类别图像全保留 (默认: 500)")
    parser.add_argument("--background", type=int, default=500,
                        help="并入训练集的无缺陷背景图数量上限 (默认: 500, 0 表示不并入)")
    parser.add_argument("--all-classes", action="store_true", help="使用全部12类 (默认仅8类主要缺陷)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (默认: 0)")

    return parser.parse_args()


def extract_zip(zip_path: str, extract_dir: str) -> str:
    """解压数据集压缩包 (zip 用 zipfile, rar 用系统 tar), 返回 EL2021 数据根目录"""
    print(f"[INFO] 正在解压: {zip_path}")
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        # RAR 归档: Windows 自带 bsdtar 可解压
        import subprocess
        ret = subprocess.run(["tar", "-xf", zip_path, "-C", extract_dir]).returncode
        if ret != 0:
            raise RuntimeError("RAR 解压失败, 请安装 WinRAR/7-Zip 后手动解压")

    # 自动向下探测 EL2021 层级目录 (其下应有 trainval/Annotations)
    current = extract_dir
    for _ in range(5):
        if os.path.isdir(os.path.join(current, "trainval")):
            return current
        subdirs = [d for d in os.listdir(current) if os.path.isdir(os.path.join(current, d))]
        if len(subdirs) == 1:
            current = os.path.join(current, subdirs[0])
        else:
            break
    raise FileNotFoundError(f"未在 {extract_dir} 下找到 EL2021 数据目录 (应包含 trainval/Annotations)")


def find_image(image_root: str, stem: str):
    """根据文件名主干查找图像文件"""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        path = os.path.join(image_root, stem + ext)
        if os.path.exists(path):
            return path
    return None


def parse_xml(xml_path: str):
    """解析 VOC XML 标注, 返回 (类别名列表, [归一化框列表])"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)

    names, boxes = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        # 转为 YOLO 归一化格式: cx, cy, w, h
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        names.append(name)
        boxes.append((cx, cy, bw, bh))

    return names, boxes


def collect_samples(src_root: str, subset: str):
    """
    扫描一个子集 (trainval/test) 的全部样本

    Returns:
        samples: [{"image": 路径, "names": 类别名列表, "boxes": 框列表}]
        class_count: {类别名: 实例数}
    """
    xml_root = os.path.join(src_root, subset, "annotations")
    if not os.path.isdir(xml_root):
        # 备选目录结构
        for cand in ("annotation", "Annotations", "xml"):
            p = os.path.join(src_root, subset, cand)
            if os.path.isdir(p):
                xml_root = p
                break
    image_root = os.path.join(src_root, subset, "images")
    if not os.path.isdir(image_root):
        for cand in ("image", "JPEGImages", "img"):
            p = os.path.join(src_root, subset, cand)
            if os.path.isdir(p):
                image_root = p
                break

    if not os.path.isdir(xml_root):
        raise FileNotFoundError(f"未找到标注目录: {os.path.join(src_root, subset)} (已尝试 annotations/annotation/Annotations/xml)")

    samples, class_count = [], defaultdict(int)
    xml_files = sorted(f for f in os.listdir(xml_root) if f.lower().endswith(".xml"))
    print(f"[INFO] 扫描 {subset} 子集: {len(xml_files)} 个 XML 标注")

    for xml_name in xml_files:
        stem = os.path.splitext(xml_name)[0]
        names, boxes = parse_xml(os.path.join(xml_root, xml_name))
        image_path = find_image(image_root, stem)
        if image_path is None:
            print(f"[WARN] 找不到对应图像, 跳过: {stem}")
            continue
        samples.append({"image": image_path, "names": names, "boxes": boxes})
        for n in names:
            class_count[n] += 1

    return samples, class_count


def balanced_sample(samples, class_count, max_per_class, rare_threshold, rng):
    """
    类别均衡采样 (仅对训练用样本执行)

    策略:
      1. 包含稀缺类 (实例数 < rare_threshold) 的图像全部保留
      2. 其余图像中, 含高频类的图像按每类上限 max_per_class 随机降采样
    """
    rare_classes = {c for c, n in class_count.items() if n < rare_threshold}
    if rare_classes:
        print(f"[INFO] 稀缺类 (实例数<{rare_threshold}): {sorted(rare_classes)}")

    # 第一轮: 保留含稀缺类的图像
    keep, rest = [], []
    seen = set()  # 防止重复计入
    for s in samples:
        if any(n in rare_classes for n in s["names"]):
            keep.append(s)
            seen.add(s["image"])
    print(f"[INFO] 含稀缺类图像全保留: {len(keep)} 张")

    # 第二轮: 高频类降采样 (逐类随机抽样, 不重复)
    per_class_pool = defaultdict(list)
    for s in samples:
        if s["image"] in seen:
            continue
        for n in set(s["names"]):
            per_class_pool[n].append(s)

    selected = set()
    for cls, pool in per_class_pool.items():
        if class_count[cls] < rare_threshold:
            continue  # 稀缺类跳过
        rng.shuffle(pool)
        quota = 0
        for s in pool:
            if quota >= max_per_class:
                break
            if s["image"] not in selected:
                selected.add(s["image"])
                quota += 1

    keep.extend(samples[i] for i, s in enumerate(samples) if s["image"] in selected)
    print(f"[INFO] 高频类采样后新增: {len(selected)} 张")
    print(f"[INFO] 均衡采样后总样本: {len(keep)} 张 (原始 {len(samples)} 张)")
    return keep


def stratified_split(samples, val_ratio, rng):
    """按图像主类分层划分 train/val"""
    groups = defaultdict(list)
    for s in samples:
        # 主类: 图像中实例数最多的类别
        main_cls = max(set(s["names"]), key=s["names"].count)
        groups[main_cls].append(s)

    train, val = [], []
    for cls, group in groups.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    print(f"[INFO] 划分结果: train={len(train)} 张, val={len(val)} 张")
    return train, val


def export_subset(samples, class_to_id, out_root, subset):
    """导出一个子集: 复制图像 + 写入 YOLO txt 标签, 返回实例统计"""
    img_dir = os.path.join(out_root, "images", subset)
    lbl_dir = os.path.join(out_root, "labels", subset)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    inst_count = defaultdict(int)
    for s in samples:
        shutil.copy2(s["image"], os.path.join(img_dir, os.path.basename(s["image"])))
        stem = os.path.splitext(os.path.basename(s["image"]))[0]
        lines = []
        for name, (cx, cy, bw, bh) in zip(s["names"], s["boxes"]):
            if name not in class_to_id:
                continue  # 排除未选类别
            lines.append(f"{class_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            inst_count[name] += 1
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return inst_count


def write_data_yaml(out_root, class_names, has_test):
    """生成 Ultralytics 数据配置文件 (path 使用绝对路径, 避免被解析到 datasets_dir)"""
    yaml_path = os.path.join(PROJECT_ROOT, "config", "pvelad.yaml")
    abs_root = os.path.abspath(out_root).replace(os.sep, "/")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# PVEL-AD 数据集配置 (由 scripts/prepare_data.py 自动生成)\n\n")
        f.write(f"path: {abs_root}\n")
        f.write(f"train: images/train\nval: images/val\n")
        if has_test:
            f.write(f"test: images/test\n")
        f.write(f"\nnames:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    print(f"[INFO] 数据配置已生成: {yaml_path}")


def print_distribution(title, count, class_to_id):
    """打印类别分布表格"""
    print(f"\n  {title}")
    print(f"  {'类别':<26} {'ID':<5} {'实例数':<8} {'占比':<8}")
    print(f"  {'-'*50}")
    total = sum(count.values()) or 1
    for name, cid in sorted(class_to_id.items(), key=lambda x: x[1]):
        n = count.get(name, 0)
        print(f"  {name:<26} {cid:<5} {n:<8} {n/total*100:<7.2f}%")


def main():
    """主流程"""
    args = parse_args()
    rng = random.Random(args.seed)

    # 1. 确定数据源目录 (EL2021 层级)
    if args.zip:
        if not os.path.exists(args.zip):
            print(f"[ERROR] 压缩包不存在: {args.zip}")
            sys.exit(1)
        extract_dir = os.path.join(os.path.dirname(args.zip), "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        src_root = extract_zip(args.zip, extract_dir)
        print(f"[INFO] 解压完成: {src_root}")
    elif args.src:
        src_root = args.src
    else:
        # 自动探测: 在 data/raw 下寻找已解压的 EL2021 目录
        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        candidates = []
        for root, dirs, _ in os.walk(raw_dir):
            if "trainval" in dirs:
                candidates.append(root)
        if not candidates:
            print("[ERROR] 未找到数据源, 请指定 --zip 或 --src 参数")
            sys.exit(1)
        src_root = candidates[0]
    print(f"[INFO] 数据源目录: {src_root}")

    # 2. 类别选择
    if args.all_classes:
        class_names = MAIN_CLASSES + RARE_CLASSES
    else:
        class_names = MAIN_CLASSES
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"[INFO] 使用类别 ({len(class_names)}类): {class_names}")

    # 3. 扫描并解析标注 (test 标注可能缺失, 容错处理)
    trainval_samples, trainval_count = collect_samples(src_root, "trainval")

    test_samples, test_count = [], {}
    try:
        test_samples, test_count = collect_samples(src_root, "test")
        if not test_samples:
            print("[WARN] test 子集无标注 (该包为旧版 EL2021), 仅使用 trainval 划分 train/val")
    except FileNotFoundError:
        print("[WARN] 未找到 test 子集, 仅使用 trainval 划分 train/val")

    print_distribution("trainval 原始类别分布", trainval_count, class_to_id)
    if test_count:
        print_distribution("test 原始类别分布", test_count, class_to_id)

    # 4. 类别均衡采样 (只作用于训练用样本)
    trainval_samples = balanced_sample(
        trainval_samples, trainval_count, args.max_per_class, args.rare_threshold, rng
    )

    # 5. 分层划分 train/val
    train_samples, val_samples = stratified_split(trainval_samples, args.val_ratio, rng)

    # 6. 无缺陷背景图并入训练集 (提升背景判别能力, 不写入标签)
    if args.background > 0:
        good_dir = os.path.join(src_root, "othertypes", "good")
        if os.path.isdir(good_dir):
            good_images = [
                os.path.join(good_dir, f) for f in os.listdir(good_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            rng.shuffle(good_images)
            picked = good_images[: args.background]
            train_samples.extend({"image": p, "names": [], "boxes": []} for p in picked)
            rng.shuffle(train_samples)
            print(f"[INFO] 并入无缺陷背景图 {len(picked)} 张 (来自 othertypes/good)")

    # 7. 导出 YOLO 格式
    out_root = os.path.join(PROJECT_ROOT, args.out)
    if os.path.isdir(out_root):
        print(f"[INFO] 清理已有输出目录: {out_root}")
        shutil.rmtree(out_root)

    print(f"\n[INFO] 导出训练集 ({len(train_samples)} 张)...")
    train_dist = export_subset(train_samples, class_to_id, out_root, "train")
    print(f"[INFO] 导出验证集 ({len(val_samples)} 张)...")
    val_dist = export_subset(val_samples, class_to_id, out_root, "val")

    has_test = bool(test_samples)
    test_dist = {}
    if has_test:
        print(f"[INFO] 导出测试集 ({len(test_samples)} 张)...")
        test_dist = export_subset(test_samples, class_to_id, out_root, "test")

    print_distribution("train 类别分布 (均衡后)", train_dist, class_to_id)
    print_distribution("val 类别分布", val_dist, class_to_id)
    if has_test:
        print_distribution("test 类别分布", test_dist, class_to_id)

    # 8. 生成数据配置
    write_data_yaml(out_root, class_names, has_test)

    print(f"\n[完成] 数据准备完成! 输出目录: {out_root}")
    print(f"[提示] 开始训练: python train.py --model yolov8n")


if __name__ == "__main__":
    main()
