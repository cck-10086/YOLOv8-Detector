"""
工具模块 - 提供模型下载、配置加载、类别映射等辅助功能
"""

import os
import sys
import yaml
import requests
from pathlib import Path
from typing import Dict, Optional, List


# COCO 数据集 80 个类别名称（中英文对照）
COCO_CLASSES = {
    0:  "person (人)",
    1:  "bicycle (自行车)",
    2:  "car (汽车)",
    3:  "motorcycle (摩托车)",
    4:  "airplane (飞机)",
    5:  "bus (公共汽车)",
    6:  "train (火车)",
    7:  "truck (卡车)",
    8:  "boat (船)",
    9:  "traffic light (红绿灯)",
    10: "fire hydrant (消防栓)",
    11: "stop sign (停止标志)",
    12: "parking meter (停车计时器)",
    13: "bench (长椅)",
    14: "bird (鸟)",
    15: "cat (猫)",
    16: "dog (狗)",
    17: "horse (马)",
    18: "sheep (羊)",
    19: "cow (牛)",
    20: "elephant (大象)",
    21: "bear (熊)",
    22: "zebra (斑马)",
    23: "giraffe (长颈鹿)",
    24: "backpack (背包)",
    25: "umbrella (雨伞)",
    26: "handbag (手提包)",
    27: "tie (领带)",
    28: "suitcase (手提箱)",
    29: "frisbee (飞盘)",
    30: "skis (滑雪板)",
    31: "snowboard (滑雪板)",
    32: "sports ball (运动球)",
    33: "kite (风筝)",
    34: "baseball bat (棒球棒)",
    35: "baseball glove (棒球手套)",
    36: "skateboard (滑板)",
    37: "surfboard (冲浪板)",
    38: "tennis racket (网球拍)",
    39: "bottle (瓶子)",
    40: "wine glass (酒杯)",
    41: "cup (杯子)",
    42: "fork (叉子)",
    43: "knife (刀)",
    44: "spoon (勺子)",
    45: "bowl (碗)",
    46: "banana (香蕉)",
    47: "apple (苹果)",
    48: "sandwich (三明治)",
    49: "orange (橙子)",
    50: "broccoli (西兰花)",
    51: "carrot (胡萝卜)",
    52: "hot dog (热狗)",
    53: "pizza (披萨)",
    54: "donut (甜甜圈)",
    55: "cake (蛋糕)",
    56: "chair (椅子)",
    57: "couch (沙发)",
    58: "potted plant (盆栽)",
    59: "bed (床)",
    60: "dining table (餐桌)",
    61: "toilet (马桶)",
    62: "tv (电视)",
    63: "laptop (笔记本电脑)",
    64: "mouse (鼠标)",
    65: "remote (遥控器)",
    66: "keyboard (键盘)",
    67: "cell phone (手机)",
    68: "microwave (微波炉)",
    69: "oven (烤箱)",
    70: "toaster (烤面包机)",
    71: "sink (水槽)",
    72: "refrigerator (冰箱)",
    73: "book (书)",
    74: "clock (时钟)",
    75: "vase (花瓶)",
    76: "scissors (剪刀)",
    77: "teddy bear (泰迪熊)",
    78: "hair drier (吹风机)",
    79: "toothbrush (牙刷)",
}


def load_config(config_path: str) -> Dict:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_class_name(class_id: int, lang: str = "zh") -> str:
    """
    根据类别 ID 获取类别名称

    Args:
        class_id: 类别 ID (0-79)
        lang: 语言，'zh' 中英文，'en' 纯英文

    Returns:
        类别名称字符串
    """
    name = COCO_CLASSES.get(class_id, f"class_{class_id} (未知)")
    if lang == "en":
        # 提取英文部分
        name = name.split(" (")[0]
    return name


def ensure_model(model_name: str, model_dir: str = "models") -> str:
    """
    确保模型文件存在，如果不存在则自动从官方渠道下载

    Args:
        model_name: 模型名称，如 "yolov8n"
        model_dir: 模型存放目录

    Returns:
        模型文件的完整路径
    """
    os.makedirs(model_dir, exist_ok=True)

    # 确保模型名称包含 .pt 扩展名
    if not model_name.endswith(".pt"):
        model_filename = f"{model_name}.pt"
    else:
        model_filename = model_name

    model_path = os.path.join(model_dir, model_filename)

    # 如果模型已存在，直接返回
    if os.path.exists(model_path):
        print(f"[INFO] 模型已存在: {model_path}")
        return model_path

    # 自动下载模型
    print(f"[INFO] 模型不存在，开始从官方渠道下载: {model_filename}")
    download_model(model_filename, model_path)

    return model_path


def download_model(model_filename: str, save_path: str) -> None:
    """
    从 Ultralytics 官方 GitHub 仓库下载预训练模型

    Args:
        model_filename: 模型文件名（如 yolov8n.pt）
        save_path: 保存路径
    """
    base_url = "https://github.com/ultralytics/assets/releases/latest/download"

    # 提取不含扩展名的模型名称
    model_name, ext = os.path.splitext(model_filename)
    url = f"{base_url}/{model_filename}"

    print(f"[INFO] 下载地址: {url}")

    try:
        # 如果 ultralytics 包已安装，优先使用其内置下载功能
        from ultralytics import YOLO

        print(f"[INFO] 使用 ultralytics 包自动下载模型...")
        # YOLO(model_name) 会自动触发下载预训练模型
        model = YOLO(f"{model_name}.pt")

        # 若下载到了当前目录，移动到目标 models/ 目录
        default_path = os.path.join(os.getcwd(), model_filename)
        if os.path.exists(default_path) and os.path.abspath(default_path) != os.path.abspath(save_path):
            import shutil
            shutil.move(default_path, save_path)
            print(f"[INFO] 模型已移动到: {save_path}")
        print(f"[INFO] 模型下载成功!")

    except Exception as e:
        print(f"[WARN] ultralytics 自动下载失败 ({e})，尝试使用 requests 下载...")
        _download_with_requests(model_filename, save_path, base_url)


def _download_with_requests(model_filename: str, save_path: str, base_url: str) -> None:
    """
    使用 requests 库下载模型文件

    Args:
        model_filename: 模型文件名
        save_path: 保存路径
        base_url: 下载基础 URL
    """
    url = f"{base_url}/{model_filename}"

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = downloaded / total_size * 100
                print(f"\r[INFO] 下载进度: {progress:.1f}% ({downloaded}/{total_size} bytes)", end="")

    print(f"\n[INFO] 模型下载成功! 已保存至: {save_path}")


def auto_download_models(model_names: Optional[list] = None, model_dir: str = "models") -> Dict[str, str]:
    """
    批量自动下载 YOLOv8 预训练模型

    Args:
        model_names: 模型名称列表，默认下载 yolov8n 和 yolov8s
        model_dir: 模型存放目录

    Returns:
        {模型名: 模型路径} 字典
    """
    if model_names is None:
        model_names = ["yolov8n", "yolov8s"]

    model_paths = {}
    for name in model_names:
        path = ensure_model(name, model_dir)
        model_paths[name] = path

    return model_paths


def is_image_file(filepath: str) -> bool:
    """判断是否为图像文件"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return os.path.splitext(filepath)[1].lower() in image_extensions


def is_video_file(filepath: str) -> bool:
    """判断是否为视频文件"""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    return os.path.splitext(filepath)[1].lower() in video_extensions


def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_onnx(model_name: str = "yolov8n", model_dir: str = "models") -> str:
    """
    将 YOLOv8 .pt 模型导出为 ONNX 格式，用于 OpenCV-DNN 轻量化部署

    Args:
        model_name: 模型名称 (如 yolov8n)
        model_dir: 模型目录

    Returns:
        onnx 文件绝对路径
    """
    pt_path = ensure_model(model_name, model_dir)
    onnx_path = os.path.splitext(pt_path)[0] + ".onnx"

    if os.path.exists(onnx_path):
        print(f"[INFO] ONNX 模型已存在: {onnx_path}")
        return onnx_path

    print(f"[INFO] 开始导出 ONNX 模型: {pt_path} -> {onnx_path}")
    print(f"[INFO] 首次导出需要安装 onnx 依赖，可能需要 1-2 分钟...")

    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        # 导出为 ONNX，使用 opset=12（与 OpenCV 兼容性最好），简化后处理
        exported = model.export(format="onnx", opset=12, simplify=True, imgsz=640)
        print(f"[INFO] ONNX 导出成功: {exported}")
        return str(exported)
    except Exception as e:
        print(f"[ERROR] ONNX 导出失败: {e}")
        print(f"[提示] 请先安装依赖: pip install onnx onnxruntime")
        raise


def load_coco_names(names_path: str = None) -> List[str]:
    """
    加载 COCO 类别名称列表

    Args:
        names_path: coco.names 文件路径，None 时使用默认路径 config/coco.names

    Returns:
        类别名称列表
    """
    if names_path is None:
        names_path = os.path.join(get_project_root(), "config", "coco.names")

    if not os.path.exists(names_path):
        # 如果没有 coco.names 文件，从 COCO_CLASSES 字典生成
        return [COCO_CLASSES[i].split(" (")[0] for i in range(len(COCO_CLASSES))]

    with open(names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

