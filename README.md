# YOLOv8 目标检测项目 🔍

基于 **Ultralytics YOLOv8** 框架的目标检测系统，支持图像/视频/摄像头实时目标检测、ONNX 轻量化部署、性能对比测试和结果可视化。

在通用检测能力之上，本项目扩展了**光伏电池 EL 图像缺陷检测**模块（课程项目：基于YOLOv8的光伏电池片EL图像缺陷检测方法研究与应用），基于 PVEL-AD 数据集实现缺陷检测模型的训练、评估与注意力机制改进。

---

## 📁 项目结构

```
YOLOv8-Detector/
├── config/
│   ├── config.yaml              # 配置文件（模型、输入输出、可视化参数）
│   ├── coco.names               # COCO 80 类别名称（用于 OpenCV-DNN）
│   ├── pvelad.yaml              # PVEL-AD 缺陷检测数据集配置（prepare_data.py 生成）
│   └── yolov8n-ca.yaml          # 改进模型结构（嵌入坐标注意力 CoordAtt）
├── models/                      # 模型目录（.pt/.onnx 运行时自动下载）
├── data/
│   ├── raw/                     # 原始数据集（PVEL-AD.zip）
│   └── PVEL-AD/                 # 准备好的 YOLO 格式数据（images/ + labels/）
├── scripts/
│   └── prepare_data.py          # 数据准备（XML→YOLO转换、类别均衡采样、数据集划分）
├── inputs/
│   └── images/
│       └── test_sample.jpg      # 测试图像（停止标志）
├── outputs/                     # 检测结果输出
├── runs/
│   ├── train/                   # 训练输出（权重、日志、曲线）
│   └── val/                     # 评估输出（混淆矩阵、PR曲线）
├── src/
│   ├── detector.py              # 核心检测模块
│   ├── visualizer.py            # 可视化模块
│   ├── utils.py                 # 工具模块（模型下载、ONNX导出、类别映射）
│   └── coordatt.py              # 坐标注意力模块（CVPR 2021, 用于模型改进）
├── main.py                      # 主程序入口（8 种推理模式）
├── train.py                     # 训练入口（基线/改进模型）
├── eval.py                      # 评估入口（多模型指标对比）
├── requirements.txt             # Python 依赖列表
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 环境要求

| 依赖项 | 最低版本 | 说明 |
|--------|---------|------|
| Python | 3.8+ | |
| PyTorch | 1.8.0+ | image/video/camera/batch/download 模式 |
| OpenCV | 4.6.0+ | 所有模式都需要 |
| ONNX | 1.12.0+ | export/onnx/benchmark 模式需要 |
| CUDA (可选) | | GPU 推理加速 |

### 2. 安装依赖

```bash
cd YOLOv8-Detector
pip install -r requirements.txt
```

### 3. 下载模型（运行时会自动下载）

```bash
python main.py --mode download --model yolov8n yolov8s yolov8m yolov8l yolov8x
```

---

## ☀️ 光伏EL缺陷检测扩展（课程项目模块）

基于 [PVEL-AD 数据集](https://github.com/binyisu/PVEL-AD)（河北工业大学 & 北京航空航天大学，IEEE TII）的光伏电池片 EL 图像缺陷检测。

### 1. 数据准备

```bash
# 下载原始数据集放置到 data/raw/PVEL-AD.zip，然后一键准备
python scripts/prepare_data.py --zip data/raw/PVEL-AD.zip

# 主要步骤: 解压 -> VOC XML 转 YOLO 格式 -> 类别均衡采样 -> train/val 分层划分 -> 生成数据配置
# 默认使用 8 类主要缺陷 (finger/crack/black_core/thick_line/横纵向错位/短路/星形裂纹)
# 加 --all-classes 使用全部 12 类; --max-per-class 控制高频类降采样上限
```

### 2. 模型训练

```bash
# 基线模型 (COCO 预训练权重迁移学习)
python train.py --model yolov8n

# 改进模型 (坐标注意力注入 backbone 四个 C2f, 预训练权重全量保留) —— 推荐
python train.py --model yolov8n --ca 2,4,6,8

# 改进模型 (插入式旧方案: P3/P4/P5 检测尺度嵌入 CoordAtt, 可作消融对比)
python train.py --model config/yolov8n-ca.yaml

# 常用参数: --epochs 100 --batch 16 --imgsz 640 --device cuda:0

# 断点恢复 (中断后从 last.pt 继续, 进度不丢失)
python train.py --resume runs/train/<run名>/weights/last.pt
```

### 3. 模型评估

```bash
# 验证集评估
python eval.py --model runs/train/yolov8n/weights/best.pt

# 测试集评估 + 多模型对比
python eval.py --model runs/train/yolov8n/weights/best.pt runs/train/yolov8n-ca2/weights/best.pt --split test
```

输出指标: 精确率 P、召回率 R、mAP50、mAP50-95、推理速度；混淆矩阵与 PR 曲线保存在 `runs/val/`。

### 4. 实验结果

**数据集**：PVEL-AD（12 类缺陷、40358 个边界框），类别均衡采样后 train 4102 张 / val 898 张 / test 19150 张。

| 模型 | val mAP50 | test mAP50 | test mAP50-95 | 推理速度 | 状态 |
|------|-----------|-----------|---------------|---------|------|
| YOLOv8n 基线 | 0.882 | **0.811** | 0.515 | 4.5ms/张 | 96 epoch 早停收敛 |
| YOLOv8n-CA 插入式 v1 | 0.675 | 0.612 | 0.423 | 3.5ms/张 | 负结果（见下） |
| YOLOv8n-CA2 继承式 v2 | - | - | - | - | 训练中 |

**基线分类别表现（test）**：短路 0.994、黑芯 0.988、断栅 0.937、横向错位 0.890 已近饱和；星形裂纹 0.702、增厚栅线 0.699、纵向错位 0.643、**裂纹 0.641** 仍有提升空间——细线状低对比度缺陷正是注意力改进的目标方向。

**插入式 v1 负结果分析**：在 neck 输出处插入 4 个 CoordAtt 层导致层索引偏移，COCO 预训练权重仅迁移 186/399 项（47%），neck+head 实际从零训练，100 epoch 内两类错位缺陷完全失效（mAP50=0）。该结果说明：**注意力改进若破坏迁移学习链路，代价远大于收益**。

**继承式 v2 设计**：C2f_CA 继承原生 C2f，forward 中对每个 Bottleneck 输出施加 CA；参数名与 C2f 完全一致，通过 `inject_coord_attention()` 在构建后原地替换 backbone C2f——预训练权重 100% 保留，CA 新增参数仅 6.3K（总参数 3.16M 的 0.2%）。

### 5. 改进说明

- **坐标注意力 (CA)**：沿水平/垂直方向分别池化生成方向感知权重，增强小目标与低对比度缺陷（裂纹、断栅）的特征提取（见 `src/coordatt.py`，论文 CVPR 2021）
- **类别均衡采样**：PVEL-AD 呈长尾分布（finger 类约 2.5 万框 vs scratch 类仅 8 框），训练时对高频类降采样、稀缺类图像全保留

---

## 📖 运行模式一览

| 模式 | 功能 | 推理引擎 | 适用场景 |
|------|------|---------|---------|
| `image` | 单张图像检测 | PyTorch | 快速测试 |
| `video` | 视频文件检测 | PyTorch | 离线视频分析 |
| `batch` | 批量图像检测 | PyTorch | 数据集推理 |
| `camera` | 摄像头实时检测 | PyTorch | 实时监控 |
| `export` | 导出 ONNX 模型 | - | 模型转换 |
| `onnx` | OpenCV-DNN 检测 | ONNX/DNN | 轻量化部署 |
| `benchmark` | 性能对比测试 | PyTorch+ONNX | 模型评估 |
| `download` | 下载预训练模型 | - | 初始化 |

---

### 图像目标检测

```bash
# 单张图像检测
python main.py --mode image --input inputs/images/test_sample.jpg --conf 0.4

# 使用更大模型、GPU 加速
python main.py --mode image --model yolov8m --device cuda:0 --input test.jpg

# 不显示窗口，仅保存
python main.py --mode image --input inputs/images/test_sample.jpg --no-save --conf 0.4
```

### 视频目标检测

```bash
# 检测视频文件
python main.py --mode video --input inputs/videos/sample.mp4

# 实时显示 + 跳帧加速
python main.py --mode video --input inputs/videos/sample.mp4 --show --frame-skip 2
```

### 批量图像检测

```bash
python main.py --mode batch --input inputs/images --conf 0.4
```

### 摄像头实时检测

```bash
# PyTorch 版本（按 'q' 退出，按 's' 保存当前帧快照）
python main.py --mode camera --conf 0.4

# 指定摄像头 + 保存检测视频
python main.py --mode camera --camera-id 1 --save-video --conf 0.4
```

### ONNX 轻量化部署

```bash
# 第一步：导出 ONNX 模型
python main.py --mode export --model yolov8n

# 第二步：使用 ONNX 模型检测图像
python main.py --mode onnx --onnx-input inputs/images/test_sample.jpg --conf 0.4

# 或使用 ONNX 模型驱动摄像头
python main.py --mode onnx --conf 0.4
```

### 性能对比测试 (Benchmark)

```bash
# PyTorch vs OpenCV-DNN 推理速度对比
python main.py --mode benchmark --conf 0.4 --runs 30
```

输出示例：
```
  方法                           平均(ms)       最小(ms)       FPS        目标数
  ----------------------------------------------------------------------
  PyTorch (ultralytics)        38.7         5.1          25.82      1
  OpenCV-DNN (ONNX)            146.2        141.0        6.84       2
  ----------------------------------------------------------------------
  速度比 (PyTorch / ONNX): 0.26x
  -> PyTorch 版本快 3.78 倍
```

---

## 🧠 可用模型

| 模型 | 参数量 | 速度 | 精度 (mAP50-95) | 适用场景 |
|------|--------|------|-----------------|----------|
| `yolov8n` | 3.2M | ⚡极快 | 37.3 | 实时检测、边缘设备 |
| `yolov8s` | 11.2M | 🚀快 | 44.9 | 通用场景、平衡推荐 |
| `yolov8m` | 25.9M | 🔄中等 | 50.2 | 精度要求较高 |
| `yolov8l` | 43.7M | 🐢较慢 | 52.9 | 高精度需求 |
| `yolov8x` | 68.2M | 🐢慢 | 53.9 | 最高精度 |

---

## 🎯 检测目标类别（COCO 80类）

| 类别 | 示例 |
|------|------|
| 人物 | person |
| 交通工具 | car, bicycle, motorcycle, bus, truck, airplane, boat, train |
| 动物 | cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe |
| 食物 | apple, banana, orange, sandwich, pizza, donut, cake, carrot, broccoli |
| 家具 | chair, couch, bed, dining table, potted plant, tv |
| 电子产品 | laptop, cell phone, keyboard, mouse, remote, microwave, oven, refrigerator |
| 生活用品 | backpack, umbrella, handbag, suitcase, bottle, cup, bowl, book, clock |

---

## ⚙️ 配置说明

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

## 🔧 Python API 调用

### PyTorch 检测器

```python
from src.detector import YOLODetector
from src.visualizer import Visualizer

detector = YOLODetector(model_name="yolov8s", device="cpu", conf_threshold=0.5)

# 检测图像
result = detector.detect_image("test.jpg", save_result=True)
print(f"检测到 {result['num_objects']} 个目标, 耗时 {result['inference_time_ms']:.1f}ms")

# 检测视频
result = detector.detect_video("test.mp4", save_dir="outputs/videos")

# 批量检测
results = detector.detect_batch("inputs/images")

# 摄像头实时检测
result = detector.detect_camera(camera_id=0)

# 打印结果表格
viz = Visualizer()
viz.print_detections_table(result['detections'])
```

### OpenCV-DNN 轻量化检测器

```python
from src.detector import OptimizedDetector

detector = OptimizedDetector(
    model_path="models/yolov8n.onnx",
    names_path="config/coco.names",
    conf_threshold=0.5,
    nms_threshold=0.4,
)

# 检测图像
result = detector.detect_image("test.jpg", save_result=True)

# 摄像头实时检测
result = detector.detect_camera(camera_id=0)
```

### ONNX 模型导出

```python
from src.utils import export_onnx
onnx_path = export_onnx(model_name="yolov8n", model_dir="models")
print(f"ONNX 模型已导出: {onnx_path}")
```

---

## 📊 返回数据结构

```python
# YOLODetector 返回结构
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

# OptimizedDetector 返回结构
{
    "detections": [
        {
            "class_id": 11,
            "class_name": "stop sign",
            "confidence": 0.496,
            "bbox": [502, 245, 573, 316]
        }
    ],
    "num_objects": 1,
    "inference_time_ms": 188.3
}
```

---

## 📊 测试结果

| 测试项 | 图像检测 | ONNX检测 | 批量检测 | 性能对比 |
|--------|---------|---------|---------|---------|
| 运行状态 | ✅ 通过 | ✅ 通过 | ✅ 通过 | ✅ 通过 |
| 检测目标 | 1个(停止标志) | 2个 | 1个 | |
| 置信度 | 0.496 | - | 0.496 | |
| 推理耗时 | 604ms | 188ms | 455ms | 38.7ms(PyTorch) |

---

## ❓ 常见问题

**Q: 默认置信度 0.5 检测不到目标？**
A: 测试图像中的停止标志置信度为 0.496，略低于 0.5。建议使用 `--conf 0.4` 或更换更大模型。

**Q: ONNX 模式报错找不到模型？**
A: 先执行 `python main.py --mode export --model yolov8n` 导出 ONNX 模型。

**Q: 摄像头无法打开？**
A: Windows 下检查摄像头权限；无摄像头时可使用 `--onnx-input` 指定图像文件。

**Q: GPU 推理报错？**
A: 确认 PyTorch 版本与 CUDA 版本匹配：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: ONNX 比 PyTorch 慢？**
A: CPU 环境下 PyTorch 内部已对 ultralytics 做深度优化。ONNX 优势在边缘设备部署（无需安装 PyTorch），或在 CUDA 环境下使用 `DNN_BACKEND_CUDA` 加速。

---

## 🔗 参考链接

- [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com)
- [YOLOv8 GitHub 仓库](https://github.com/ultralytics/ultralytics)
- [OpenCV DNN 模块文档](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [COCO 数据集](https://cocodataset.org)
- [ONNX 格式说明](https://onnx.ai)
