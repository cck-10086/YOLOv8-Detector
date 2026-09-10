# YOLOv8 目标检测项目 🔍

基于 **Ultralytics YOLOv8** 框架的目标检测系统，支持图像/视频/摄像头实时目标检测、ONNX 轻量化部署、性能对比测试和结果可视化。

---

## 📁 项目结构

```
Chapter06/
├── config/
│   ├── config.yaml              # 配置文件（模型、输入输出、可视化参数）
│   └── coco.names               # COCO 80 类别名称（用于 OpenCV-DNN）
├── models/                      # 模型目录（.pt/.onnx 运行时自动下载）
│   ├── yolov8n.pt               # YOLOv8 nano 预训练模型 (PyTorch)
│   └── yolov8n.onnx             # ONNX 导出模型（轻量化部署用）
├── inputs/
│   └── images/
│       └── test_sample.jpg      # 测试图像（停止标志）
├── outputs/
│   ├── images/                  # 图像检测结果输出
│   └── videos/                  # 视频检测结果输出
├── src/
│   ├── __init__.py              # 包初始化
│   ├── detector.py              # 核心检测模块
│   │   ├── YOLODetector         #   PyTorch 推理检测器
│   │   └── OptimizedDetector    #   OpenCV-DNN 轻量化检测器 (ONNX)
│   ├── visualizer.py            # 可视化模块（边界框、标签、置信度、对比图）
│   └── utils.py                 # 工具模块（模型下载、ONNX导出、类别映射）
├── main.py                      # 主程序入口（8 种运行模式）
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
cd Chapter06
pip install -r requirements.txt
```

### 3. 下载模型（运行时会自动下载）

```bash
python main.py --mode download --model yolov8n yolov8s yolov8m yolov8l yolov8x
```

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
