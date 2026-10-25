# 基于 YOLOv8 的光伏电池片 EL 图像缺陷检测

《计算机视觉与图像处理综合实践》课程论文配套代码 · 作者 人工智能学院 计科2301 崔程柯（2023500328） · 指导教师 陈浩

以 **YOLOv8n** 为基线模型，向主干网络嵌入**坐标注意力（Coordinate Attention, CA）**构成的 `C2f_CA` 模块，
在 PVEL-AD 数据集的 8 类光伏电池片缺陷上完成训练、对比实验、消融实验与 ONNX 轻量化部署。
项目同时保留了一套通用 YOLOv8 检测工具链（图像 / 视频 / 摄像头 / 批量推理、ONNX 部署、性能对比）。

论文交付稿见 [`../docs/基于YOLOv8的光伏电池片EL图像缺陷检测方法研究与应用.docx`](../docs/)。

---

## 一、项目结构

```
YOLOv8-Detector/
├── config/
│   ├── config.yaml              # 通用检测配置（模型、输入输出、可视化参数）
│   ├── coco.names               # COCO 80 类别名称（OpenCV-DNN 通用检测用）
│   ├── pvelad.yaml              # PVEL-AD 缺陷检测数据集配置（prepare_data.py 生成）
│   └── yolov8n-ca.yaml          # 颈部插入式对照模型结构（消融实验用）
├── src/
│   ├── detector.py              # 核心检测模块（PyTorch / OpenCV-DNN 两种实现）
│   ├── visualizer.py            # 检测结果可视化与表格输出
│   ├── utils.py                 # 模型下载、ONNX 导出、类别映射等工具
│   └── coordatt.py              # 坐标注意力与 C2f_CA 模块、注入函数
├── scripts/
│   ├── prepare_data.py          # XML→YOLO 数据转换、类别均衡采样、数据集划分
│   ├── visualize.py             # 一键生成论文插图（fig1~fig4）
│   ├── export_onnx.py           # ONNX 导出 + 双后端精度一致性验证
│   └── onnx_infer.py            # 纯 onnxruntime 独立推理演示（不依赖 PyTorch）
├── experiments/                 # 实验编排、评估与诊断脚本
│   ├── train_all_ca.py          # 批量训练 CA 变体（产出 ca-all-real / ca-deep-real2 / ca-shallow-real）
│   ├── train_shallow_only.py    # 浅层 CA 单独训练
│   ├── train_deep.py            # 深层 CA 训练
│   ├── train_verify.py          # 训练链路验证
│   ├── val_class_ap.py          # val 集分类别 AP 评估（写入 results/val_metrics.json）
│   ├── val_shallow_only.py      # 浅层模型单独评估并合并结果
│   ├── test_eval.py             # test 集整体 + 分类别评估
│   ├── extract_metrics.py       # 从 results.csv 汇总指标
│   ├── count_params.py          # 统计各模型参数量
│   ├── diagnose_modules.py      # 检查模型内 CA 模块分布
│   ├── diagnose_ca.py           # CA 注入诊断
│   └── raw_inspect.py           # 绕过封装直接 torch.load 检查真实层类型
├── models/
│   └── best.onnx                # ONNX 部署模型（scripts/onnx_infer.py 默认加载）
├── data/
│   ├── raw/                     # 原始数据集（PVEL-AD.zip，约 4.2GB，已 gitignore）
│   └── PVEL-AD/                 # 准备好的 YOLO 格式数据（images/ + labels/）
├── inputs/images/test_sample.jpg
├── outputs/                     # 通用检测结果输出
├── results/
│   ├── figures/                 # 论文插图（fig1~fig4）
│   ├── val_metrics.json         # val 集统一口径指标（论文表4-1、表4-2 的数据来源）
│   └── onnx_deploy_demo.jpg     # 部署测试结果图（论文图5-1）
├── runs/
│   ├── train/                   # 各模型训练输出（权重、日志、曲线）
│   ├── val/                     # 评估输出（混淆矩阵、PR 曲线）
│   └── val_eval/
├── _archive/                    # 归档区（过程日志、早期产物，可安全删除）
├── main.py                      # 通用检测入口（8 种推理模式）
├── train.py                     # 训练入口（基线 / CA 改进 / 插入式对照）
├── eval.py                      # 评估入口（多模型指标对比）
├── requirements.txt
└── README.md
```

---

## 二、环境与依赖

论文实验环境：

| 项目 | 配置 | 项目 | 配置 |
|------|------|------|------|
| 操作系统 | Windows 11 | CUDA | 12.4 |
| CPU | Intel Core | 编程语言 | Python 3.8 |
| GPU | RTX 4060 Laptop 8GB | 深度学习框架 | PyTorch 2.4.1 |
| 检测框架 | Ultralytics 8.4.33 | 加速库 | cuDNN |

```bash
cd YOLOv8-Detector
pip install -r requirements.txt
```

---

## 三、数据准备

采用公开数据集 [PVEL-AD](https://github.com/binyisu/PVEL-AD)（Su 等，IEEE TII 2023）。
本项目使用其中 8 类典型缺陷：断栅、裂纹、黑芯、增厚栅线、横向错位、短路、纵向错位、星形裂纹。

```bash
# 下载原始数据集放置到 data/raw/PVEL-AD.zip，然后一键准备
python scripts/prepare_data.py --zip data/raw/PVEL-AD.zip

# 流程: 解压 → VOC XML 转 YOLO 格式 → 类别均衡采样 → train/val 分层划分 → 生成数据配置
# 默认 8 类主要缺陷；--all-classes 使用全部 12 类；--max-per-class 控制高频类降采样上限
```

数据集规模：训练集 4102 张（含 505 张无缺陷背景图）、验证集 898 张（含 2 张背景图）、测试集 19150 张。
训练集类别分布呈现明显长尾：断栅最多（2369 个实例），纵向错位（112 个）与星形裂纹（109 个）样本极少，
故训练时对高频类降采样、稀缺类图像全保留，并采用 Mosaic、MixUp、随机翻转缩放、HSV 扰动等增强手段。

---

## 四、模型改进

### 1. 改进思路

光伏 EL 图像中的裂纹、断栅、错位等缺陷具有明显方向性与细长结构，且部分缺陷对比度低。
标准 `C2f` 模块虽能高效提取特征，却缺乏对空间位置信息的显式建模。
本文遵循两条原则：注意力应尽可能靠近特征提取过程、与 `C2f` 融合；不破坏检测颈部的多尺度特征融合结构。

### 2. C2f_CA 模块

`C2f_CA` 保持 `C2f` 的多分支结构不变，仅在**分支特征拼接与融合之后**追加坐标注意力子模块：
对融合特征分别沿水平、垂直方向做一维全局池化，生成两个方向的方向感知权重并作用于融合特征。

注入方式为**继承式替换**：`C2f_CA` 继承原生 `C2f`，参数命名与 `C2f` 完全一致，
通过 `inject_coord_attention()` 在模型构建后原地替换主干 `C2f`，**预训练权重完整迁移**，坐标注意力以随机初始化参与训练。

YOLOv8n 主干共 4 个 `C2f`（第 2、4、6、8 层，输出通道 32/64/128/256）。
以论文最终采用的深层方案（第 6、8 层）为例，坐标注意力仅新增约 5K 参数，
总参数由 **3.007M 增至 3.012M**（增量不足 0.2%），体现了 CA 的轻量特性。

### 3. 训练

```bash
# 基线模型（COCO 预训练权重迁移学习）
python train.py --model yolov8n

# 改进模型 · 深层（第6、8层）—— 论文最终方案
python train.py --model yolov8n --ca 6,8

# 改进模型 · 全层（第2、4、6、8层）
python train.py --model yolov8n --ca 2,4,6,8

# 改进模型 · 浅层（第2、4层）
python train.py --model yolov8n --ca 2,4

# 颈部插入式旧方案（消融对照）
python train.py --model config/yolov8n-ca.yaml

# 断点恢复
python train.py --resume runs/train/<run名>/weights/last.pt
```

训练超参（基线与各改进模型完全一致，保证对比公平）：

| 超参数 | 取值 | 超参数 | 取值 |
|--------|------|--------|------|
| 输入尺寸 | 640×640 | 优化器 | SGD/AdamW (auto) |
| 训练轮数 | 100 | 初始学习率 | 0.01 |
| 批大小 | 16 | 动量 | 0.937 |
| 早停耐心值 | 25 | 权重衰减 | 0.0005 |
| 随机种子 | 0 | 混合精度 | 开启 (AMP) |

---

## 五、实验结果

以下数据为验证集口径（best.pt，与 `results/val_metrics.json` 一致），对应论文表4-1、表4-2。

**表4-1 基线模型与改进模型（深层 CA）性能对比**

| 模型 | 精确率 P/% | 召回率 R/% | mAP50/% | mAP50-95/% | 参数量/M | 推理耗时/ms |
|------|-----------|-----------|---------|------------|---------|------------|
| YOLOv8n（基线） | 79.92 | 84.17 | 88.04 | 59.81 | 3.007 | 2.9 |
| YOLOv8n + CA（深层） | **88.75** | 82.04 | **90.17** | **62.16** | 3.012 | 3.0 |

**表4-2 坐标注意力嵌入位置消融实验**

| 模型 | C2f_CA 位置 | mAP50/% | mAP50-95/% | 参数量/M |
|------|------------|---------|------------|---------|
| 基线 YOLOv8n | 无 | 88.04 | 59.81 | 3.007 |
| 浅层 CA | 第 2、4 层 | 86.41 | 57.84 | 3.009 |
| **深层 CA** | 第 6、8 层 | **90.17** | **62.16** | 3.012 |
| 全层 CA | 第 2、4、6、8 层 | 88.14 | 58.82 | 3.014 |
| 颈部插入式（对照） | 检测头前独立 CA | 67.54 | 49.26 | 3.027 |

**消融结论**

- **深层最优**：第 6、8 层注入使 mAP50 达 90.17%，较基线提升约 2.1 个百分点，mAP50-95 提升约 2.4 个百分点，
  精确率提升约 8.8 个百分点；深层特征感受野更大、高层语义更明确，方向性缺陷的位置特征在该阶段最显著。
- **浅层有害**：第 2、4 层单独注入 mAP50 仅 86.41%，低于基线约 1.6 个百分点，精确率下降约 4.2 个百分点；
  浅层特征尚不稳定、方向语义不明确，过早加权会干扰基础特征学习。
- **并非越多越好**：全层注入 mAP50 为 88.14%，与基线基本持平——浅层的负面作用抵消了深层的正面增益。
- **颈部独立插入不可取**：mAP50 仅 67.54%，独立注意力层改变了检测头所依赖的多尺度特征结构，未经充分融合即加权反而干扰检测。

**分类别表现（验证集 AP50）**

| 类别 | 基线 | 深层 CA | 变化 |
|------|------|---------|------|
| 横向错位 | 0.863 | **0.965** | +10.3 pt |
| 纵向错位 | 0.769 | **0.924** | +15.5 pt |
| 星形裂纹 | 0.838 | 0.845 | +0.7 pt |
| 黑芯 | 0.988 | 0.991 | +0.3 pt |
| 断栅 | 0.936 | 0.929 | -0.7 pt |
| 短路 | 0.995 | 0.995 | 0.0 pt |
| 增厚栅线 | 0.913 | 0.845 | -6.8 pt |
| 裂纹 | 0.742 | 0.720 | -2.2 pt |

坐标注意力对**方向性缺陷**（横向/纵向错位）增益最显著；已近饱和的类别（短路、黑芯、断栅）保持高精度；
增厚栅线与裂纹精度略有下降，反映深层注意力在增强方向特征时对部分类别存在特征取舍。

**论文模型 ↔ 训练目录对应关系**

| 论文中的模型 | runs/train 目录 | 备注 |
|-------------|----------------|------|
| 基线 YOLOv8n | `baseline_yolov8n/` | — |
| 颈部插入式（对照） | `train3/` | 负结果 |
| 全层 CA(2,4,6,8) | `ca-all-real/` | 消融 |
| 浅层 CA(2,4) | `ca-shallow-real/` | 消融 |
| 深层 CA(6,8) | `ca-deep-real2/` | 论文最终方案 |

> 早期实验目录（`train5`、`train6`、`ca-deep`、`ca-deep-real`、`verify-ca`）已归档到 `runs/train/_archive/`。

---

## 六、模型评估

```bash
# 验证集评估（默认 val）
python eval.py --model runs/train/ca-deep-real2/weights/best.pt

# 测试集评估 + 多模型对比
python eval.py --model runs/train/baseline_yolov8n/weights/best.pt \
                        runs/train/ca-deep-real2/weights/best.pt --split test
```

输出指标：精确率 P、召回率 R、mAP50、mAP50-95、推理速度；混淆矩阵与 PR 曲线保存在 `runs/val/`。

---

## 七、论文插图生成

```bash
python scripts/visualize.py     # 一键生成全部论文插图 -> results/figures/
```

| 图片文件 | 论文图题 |
|---------|---------|
| `fig1_training_curves.png` | 图4-1 基线模型与改进模型的训练过程对比（五模型 mAP 曲线 + 基线/深层 CA 的 box、cls 损失曲线） |
| `fig2_class_ap.png` | 图4-2 基线模型与改进模型分类别 AP50 对比（横向/纵向错位提升以箭头标注） |
| `fig3_class_distribution.png` | 图3-1 训练集各类缺陷实例数量分布（长尾形态） |
| `fig4_detection_compare.png` | 图4-3 基线模型与改进模型检测结果对比（3 组样例，改进模型置信度更高） |

---

## 八、ONNX 部署

```bash
# 导出 ONNX + 双后端精度一致性验证（PyTorch vs onnxruntime）
python scripts/export_onnx.py

# 纯 onnxruntime 独立推理演示（无 PyTorch / Ultralytics 依赖，中文标签可视化）
python scripts/onnx_infer.py --conf 0.25
```

`scripts/onnx_infer.py` 默认加载 `models/best.onnx`，可用 `--onnx` 指定其他模型。

部署验证结论：

- 导出数值无损：随机输入下双后端输出张量最大绝对误差约 1e-3（float32 精度内）；
  验证集 mAP50 双后端偏差约 0.0077（<1%，源于 NMS 边界效应，非导出损失）。
- 独立运行：仅依赖 `onnxruntime + numpy + opencv + pillow`，模拟产线部署环境，可在 CPU 上完成单张图像实时检测。
- 部署结果与原模型一致，论文图5-1（`results/onnx_deploy_demo.jpg`）为实际部署推理结果。

---

## 九、通用检测工具链（main.py）

| 模式 | 功能 | 推理引擎 | 适用场景 |
|------|------|---------|---------|
| `image` | 单张图像检测 | PyTorch | 快速测试 |
| `video` | 视频文件检测 | PyTorch | 离线视频分析 |
| `batch` | 批量图像检测 | PyTorch | 数据集推理 |
| `camera` | 摄像头实时检测 | PyTorch | 实时监控 |
| `export` | 导出 ONNX 模型 | — | 模型转换 |
| `onnx` | OpenCV-DNN 检测 | ONNX/DNN | 轻量化部署 |
| `benchmark` | 性能对比测试 | PyTorch + ONNX | 模型评估 |
| `download` | 下载预训练模型 | — | 初始化 |

```bash
# 单张图像检测（置信度阈值建议 0.4，见「常见问题」）
python main.py --mode image --input inputs/images/test_sample.jpg --conf 0.4

# 批量检测
python main.py --mode batch --input inputs/images --conf 0.4

# 视频检测（实时显示 + 跳帧加速）
python main.py --mode video --input inputs/videos/sample.mp4 --show --frame-skip 2

# 摄像头实时检测（按 q 退出，按 s 保存当前帧快照）
python main.py --mode camera --conf 0.4

# 导出 ONNX 并用 OpenCV-DNN 推理
python main.py --mode export --model yolov8n
python main.py --mode onnx --onnx-input inputs/images/test_sample.jpg --conf 0.4

# PyTorch 与 OpenCV-DNN 推理速度对比
python main.py --mode benchmark --conf 0.4 --runs 30

# 仅下载预训练模型
python main.py --mode download --model yolov8n yolov8s yolov8m yolov8l yolov8x
```

---

## 十、Python API

```python
from src.detector import YOLODetector
from src.visualizer import Visualizer

detector = YOLODetector(model_name="yolov8s", device="cpu", conf_threshold=0.5)

result = detector.detect_image("test.jpg", save_result=True)
print(f"检测到 {result['num_objects']} 个目标, 耗时 {result['inference_time_ms']:.1f}ms")

result = detector.detect_video("test.mp4", save_dir="outputs/videos")
results = detector.detect_batch("inputs/images")
result = detector.detect_camera(camera_id=0)

Visualizer().print_detections_table(result["detections"])
```

```python
# OpenCV-DNN 轻量化检测器
from src.detector import OptimizedDetector

detector = OptimizedDetector(
    model_path="models/best.onnx",
    names_path="config/coco.names",
    conf_threshold=0.5,
    nms_threshold=0.4,
)
result = detector.detect_image("test.jpg", save_result=True)
```

```python
# ONNX 模型导出
from src.utils import export_onnx
onnx_path = export_onnx(model_name="yolov8n", model_dir="models")
```

`YOLODetector.detect_image` 返回结构：

```python
{
    "image_path": "inputs/images/test_sample.jpg",
    "detections": [
        {
            "class_id": 11,
            "class_name": "stop sign (停止标志)",
            "class_name_en": "stop sign",
            "confidence": 0.496,
            "bbox": [502.0, 245.0, 573.0, 316.0],
            "bbox_norm": [0.8398, 0.5844, 0.1109, 0.1479]
        }
    ],
    "num_objects": 1,
    "inference_time_ms": 604.2,
    "saved_path": "outputs/images/test_sample_detected.jpg"
}
```

---

## 十一、配置说明

编辑 [config/config.yaml](config/config.yaml)：

```yaml
model:
  name: "yolov8n"        # 模型名称
  device: "cpu"          # 推理设备：cpu / cuda:0
  conf_threshold: 0.5    # 置信度阈值
  iou_threshold: 0.45    # NMS IoU 阈值

visualization:
  box_thickness: 2       # 边界框粗细
  font_size: 12          # 标签字体大小
  show_confidence: true  # 是否显示置信度
```

---

## 十二、常见问题

**Q：默认置信度 0.5 检测不到目标？**
A：测试图像中的停止标志置信度为 0.496，略低于 0.5，建议使用 `--conf 0.4` 或改用更大的模型。

**Q：ONNX 模式报错找不到模型？**
A：先执行 `python main.py --mode export --model yolov8n` 导出；
论文的部署模型已随项目提供，位于 `models/best.onnx`。

**Q：摄像头无法打开？**
A：Windows 下检查摄像头权限；无摄像头时可用 `--onnx-input` 指定图像文件。

**Q：GPU 推理报错？**
A：确认 PyTorch 与 CUDA 版本匹配：`python -c "import torch; print(torch.cuda.is_available())"`。

**Q：ONNX 推理比 PyTorch 慢？**
A：CPU 环境下 ultralytics 的 PyTorch 后端已做深度优化。ONNX 的优势在于边缘设备部署（无需安装 PyTorch），
或在 CUDA 环境下改用 `DNN_BACKEND_CUDA` 加速。

---

## 参考链接

- [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com)
- [YOLOv8 GitHub 仓库](https://github.com/ultralytics/ultralytics)
- [PVEL-AD 数据集](https://github.com/binyisu/PVEL-AD)
- [Coordinate Attention 论文（CVPR 2021）](https://arxiv.org/abs/2103.02907)
- [OpenCV DNN 模块文档](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
