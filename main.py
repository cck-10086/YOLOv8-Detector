#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测项目 - 主程序入口

功能:
  1. 自动下载 YOLOv8 预训练模型
  2. 图像目标检测
  3. 视频目标检测
  4. 批量图像检测
  5. 检测结果可视化

用法:
  # 检测单张图像
  python main.py --mode image --input inputs/images/sample.jpg

  # 检测视频
  python main.py --mode video --input inputs/videos/sample.mp4

  # 批量检测图像目录
  python main.py --mode batch --input inputs/images

  # 指定模型
  python main.py --mode image --model yolov8s --input inputs/images/sample.jpg

  # 仅下载模型
  python main.py --mode download --model yolov8n yolov8s

  # 使用 GPU 推理
  python main.py --mode image --device cuda:0 --input inputs/images/sample.jpg
"""

import os
import sys
import argparse
import cv2

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 目标检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode image --input inputs/images/sample.jpg
  python main.py --mode video --input inputs/videos/sample.mp4 --show
  python main.py --mode batch --input inputs/images
  python main.py --mode download --model yolov8n yolov8s
  python main.py --mode image --model yolov8m --device cuda:0 --input test.jpg
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["image", "video", "batch", "camera", "onnx", "benchmark", "export", "download"],
        help="运行模式: image, video, batch, camera(摄像头), onnx(OpenCV-DNN), benchmark(性能对比), export(导出ONNX), download(下载模型)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入文件/目录路径（image/video/batch 模式必需）",
    )

    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=["yolov8n"],
        help="模型名称，可选: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (默认: yolov8n)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备 (默认: cpu, 可选: cuda:0, cuda:1 等)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="置信度阈值 (默认: 0.5)",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU 阈值 (默认: 0.45)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="是否显示检测结果窗口（image 模式默认显示，video/batch 模式默认不显示）",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="不保存检测结果",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="自定义输出目录",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="模型存放目录 (默认: models)",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="视频检测跳帧间隔 (默认: 1，每帧都检测)",
    )

    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头 ID (默认: 0)",
    )

    parser.add_argument(
        "--onnx-input",
        type=str,
        default=None,
        help="ONNX 模式下的图像输入路径（替代摄像头）",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="benchmark 模式的迭代次数 (默认: 30)",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help="camera 模式是否保存检测视频 (默认: 不保存)",
    )

    return parser.parse_args()


def run_image_mode(args):
    """运行单张图像检测模式"""
    from src.detector import YOLODetector
    from src.visualizer import Visualizer
    from src.utils import load_config

    detector = YOLODetector(
        model_name=args.model[0],
        model_dir=args.model_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    output_dir = args.output or "outputs/images"
    show = not args.no_save or args.show  # image 模式默认显示

    result = detector.detect_image(
        image_path=args.input,
        save_result=not args.no_save,
        save_dir=output_dir,
        show_result=show,
    )

    # 打印检测结果表格
    visualizer = Visualizer()
    visualizer.print_detections_table(result["detections"])

    return result


def run_video_mode(args):
    """运行视频检测模式"""
    from src.detector import YOLODetector

    detector = YOLODetector(
        model_name=args.model[0],
        model_dir=args.model_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    output_dir = args.output or "outputs/videos"

    result = detector.detect_video(
        video_path=args.input,
        save_dir=output_dir,
        show_result=args.show,
        frame_skip=args.frame_skip,
    )

    return result


def run_batch_mode(args):
    """运行批量图像检测模式"""
    from src.detector import YOLODetector

    detector = YOLODetector(
        model_name=args.model[0],
        model_dir=args.model_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    output_dir = args.output or "outputs/images"

    results = detector.detect_batch(
        input_dir=args.input,
        save_dir=output_dir,
        show_result=args.show,
    )

    return results


def run_download_mode(args):
    """运行模型下载模式"""
    from src.utils import auto_download_models

    print(f"\n[下载] 开始下载模型: {args.model}")
    model_paths = auto_download_models(
        model_names=args.model,
        model_dir=args.model_dir,
    )

    print(f"\n[下载] 完成！模型文件:")
    for name, path in model_paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {name}: {path} ({size_mb:.1f} MB)")

    return model_paths


def run_camera_mode(args):
    """运行摄像头实时检测模式（PyTorch 版本）"""
    from src.detector import YOLODetector

    detector = YOLODetector(
        model_name=args.model[0],
        model_dir=args.model_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    save_video = hasattr(args, "save_video") and args.save_video
    result = detector.detect_camera(
        camera_id=args.camera_id if hasattr(args, "camera_id") else 0,
        save_video=save_video,
        save_dir=args.output or "outputs/videos",
        show_result=not args.no_save,
    )
    return result


def run_onnx_mode(args):
    """运行 OpenCV-DNN 轻量化检测模式"""
    from src.detector import OptimizedDetector

    model_path = args.input
    if not model_path or not os.path.exists(model_path):
        # 如未指定 onnx 文件，使用默认路径
        model_path = os.path.join(args.model_dir, f"{args.model[0]}.onnx")
        if not os.path.exists(model_path):
            print(f"[ERROR] 未找到 ONNX 模型: {model_path}")
            print(f"[提示] 先执行: python main.py --mode export --model {args.model[0]}")
            sys.exit(1)

    detector = OptimizedDetector(
        model_path=model_path,
        names_path=os.path.join("config", "coco.names"),
        conf_threshold=args.conf,
        nms_threshold=args.iou,
    )

    # 如果用户指定了图像输入，则对图像进行检测；否则启动摄像头
    input_img = getattr(args, "onnx_input", None)
    if input_img and os.path.exists(input_img):
        result = detector.detect_image(input_img, save_result=not args.no_save)
    else:
        try:
            result = detector.detect_camera(camera_id=args.camera_id if hasattr(args, "camera_id") else 0)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            print(f"[提示] 请使用 --onnx-input 指定图像文件进行检测")
            sys.exit(1)
    return result


def run_export_mode(args):
    """运行 ONNX 模型导出模式"""
    from src.utils import export_onnx

    for model_name in args.model:
        print(f"\n{'='*60}")
        print(f"[导出] 正在导出 {model_name} -> ONNX")
        onnx_path = export_onnx(model_name=model_name, model_dir=args.model_dir)
        print(f"[导出] 成功: {onnx_path}")
    return True


def run_benchmark_mode(args):
    """
    性能对比模式：对比 PyTorch (YOLOv8) vs OpenCV-DNN (ONNX) 的推理速度

    测试步骤:
    1. 对同一张图像分别用两个检测器推理 N 次
    2. 计算平均推理耗时、FPS、目标数
    3. 打印对比表格
    """
    import time
    import numpy as np
    from src.detector import YOLODetector, OptimizedDetector

    input_path = args.input or "inputs/images/test_sample.jpg"
    if not os.path.exists(input_path):
        print(f"[ERROR] 找不到测试图像: {input_path}")
        sys.exit(1)

    num_runs = getattr(args, "runs", 30)

    print(f"\n{'='*60}")
    print(f"  性能对比测试 (Benchmark)")
    print(f"  测试图像: {input_path}")
    print(f"  迭代次数: {num_runs}")
    print(f"  置信度阈值: {args.conf}")
    print(f"{'='*60}")

    # --- 1. PyTorch (ultralytics) ---
    print(f"\n[1/2] 正在测试 PyTorch (ultralytics) 版本...")
    detector_pt = YOLODetector(
        model_name=args.model[0],
        model_dir=args.model_dir,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    pt_times = []
    for _ in range(num_runs):
        t0 = time.time()
        r = detector_pt.model(source=input_path, conf=args.conf, verbose=False)
        pt_times.append((time.time() - t0) * 1000)
    pt_num_objects = len(r[0].boxes) if r and r[0].boxes else 0

    # --- 2. OpenCV-DNN (ONNX) ---
    print(f"\n[2/2] 正在测试 OpenCV-DNN (ONNX) 版本...")
    onnx_path = os.path.join(args.model_dir, f"{args.model[0]}.onnx")
    if not os.path.exists(onnx_path):
        print(f"[WARN] 未找到 ONNX 模型: {onnx_path}")
        print(f"[提示] 执行: python main.py --mode export --model {args.model[0]}")
        onnx_result = None
    else:
        detector_onnx = OptimizedDetector(
            model_path=onnx_path,
            names_path=os.path.join("config", "coco.names"),
            conf_threshold=args.conf,
            nms_threshold=args.iou,
        )
        onnx_times = []
        for _ in range(num_runs):
            t0 = time.time()
            r = detector_onnx.detect(cv2.imread(input_path))
            onnx_times.append(r["inference_time_ms"])
        onnx_num_objects = r["num_objects"]
        onnx_result = {"times": onnx_times, "objects": onnx_num_objects}

    # --- 3. 打印对比表格 ---
    print(f"\n{'='*60}")
    print(f"  性能对比结果")
    print(f"{'='*60}")
    print(f"  {'方法':<28} {'平均(ms)':<12} {'最小(ms)':<12} {'FPS':<10} {'目标数':<8}")
    print(f"  {'-'*70}")

    pt_avg = np.mean(pt_times)
    pt_min = np.min(pt_times)
    pt_fps = 1000.0 / pt_avg
    print(f"  {'PyTorch (ultralytics)':<28} {pt_avg:<12.1f} {pt_min:<12.1f} {pt_fps:<10.2f} {pt_num_objects:<8}")

    if onnx_result:
        onnx_avg = np.mean(onnx_result["times"])
        onnx_min = np.min(onnx_result["times"])
        onnx_fps = 1000.0 / onnx_avg
        print(f"  {'OpenCV-DNN (ONNX)':<28} {onnx_avg:<12.1f} {onnx_min:<12.1f} {onnx_fps:<10.2f} {onnx_result['objects']:<8}")
        print(f"  {'-'*70}")
        speedup = pt_avg / onnx_avg
        print(f"  速度比 (PyTorch / ONNX): {speedup:.2f}x")
        if speedup > 1:
            print(f"  -> OpenCV-DNN 版本快 {speedup:.2f} 倍")
        else:
            print(f"  -> PyTorch 版本快 {1/speedup:.2f} 倍")

    print(f"{'='*60}\n")
    return {"pytorch": pt_times, "onnx": onnx_result}


def main():
    """主函数"""
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"  YOLOv8 目标检测系统")
    print(f"  模型: {args.model[0]} | 设备: {args.device} | 置信度: {args.conf}")
    print(f"{'#'*60}")

    # 验证输入参数
    if args.mode in ("image", "video", "batch"):
        if not args.input:
            print(f"[ERROR] {args.mode} 模式需要指定 --input 参数")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"[ERROR] 输入路径不存在: {args.input}")
            sys.exit(1)

    if args.mode == "onnx" and args.input and not args.onnx_input:
        # onnx 模式下 --input 可能是 onnx 文件，也可能不存在
        pass

    # 根据模式执行
    mode_handlers = {
        "image": run_image_mode,
        "video": run_video_mode,
        "batch": run_batch_mode,
        "download": run_download_mode,
        "camera": run_camera_mode,
        "onnx": run_onnx_mode,
        "export": run_export_mode,
        "benchmark": run_benchmark_mode,
    }

    try:
        handler = mode_handlers[args.mode]
        result = handler(args)
        print(f"\n[完成] 任务执行成功! ✓")
        return result
    except Exception as e:
        print(f"\n[ERROR] 任务执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
