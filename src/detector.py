"""
检测器模块 - YOLOv8 目标检测核心类
支持图像和视频的目标检测推理
"""

import os
import sys
import cv2
import time
from typing import List, Dict, Optional, Tuple, Union

import numpy as np

# 将 ultralytics 框架添加到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ultralytics-main"))

from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8 目标检测器

    功能:
    - 自动下载预训练模型
    - 加载模型进行推理
    - 支持图像和视频的目标检测
    - 返回结构化的检测结果

    Usage:
        detector = YOLODetector(model_name="yolov8n", model_dir="models")
        results = detector.detect_image("path/to/image.jpg")
        detector.detect_video("path/to/video.mp4", save_path="output.mp4")
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        model_dir: str = "models",
        device: str = "cpu",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        """
        初始化 YOLOv8 检测器

        Args:
            model_name: 模型名称 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            model_dir: 模型存放目录
            device: 推理设备 ("cpu", "cuda:0", 等)
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None

        # 初始化模型
        self._load_model()

    def _load_model(self) -> None:
        """
        加载 YOLOv8 模型（自动下载如果不存在）

        Raises:
            FileNotFoundError: 模型加载失败时抛出
        """
        from src.utils import ensure_model

        # 确保模型文件存在
        model_path = ensure_model(self.model_name, self.model_dir)

        print(f"[INFO] 正在加载模型: {model_path}")
        print(f"[INFO] 推理设备: {self.device}")

        try:
            self.model = YOLO(model_path)
            # 设置设备
            if self.device != "cpu":
                self.model.to(self.device)
            print(f"[INFO] 模型加载成功! ✓")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def detect_image(
        self,
        image_path: str,
        save_result: bool = True,
        save_dir: str = "outputs/images",
        show_result: bool = True,
    ) -> Dict:
        """
        对单张图像执行目标检测

        Args:
            image_path: 输入图像路径
            save_result: 是否保存结果图像
            save_dir: 结果保存目录
            show_result: 是否显示检测结果

        Returns:
            检测结果字典，包含:
            - image_path: 原始图像路径
            - detections: 检测结果列表
            - num_objects: 检测到的目标数量
            - inference_time: 推理耗时(ms)
            - saved_path: 保存的结果路径（如果 save_result=True）
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        print(f"\n{'='*60}")
        print(f"[检测] 图像: {image_path}")

        # 执行推理
        start_time = time.time()
        results = self.model(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        inference_time = (time.time() - start_time) * 1000  # 毫秒

        # 解析结果
        detections = self._parse_results(results)

        print(f"[检测] 发现 {len(detections)} 个目标")
        print(f"[检测] 推理耗时: {inference_time:.1f} ms")

        result_dict = {
            "image_path": image_path,
            "detections": detections,
            "num_objects": len(detections),
            "inference_time_ms": inference_time,
            "saved_path": None,
        }

        # 可视化并保存结果
        if save_result and detections:
            from src.visualizer import Visualizer

            visualizer = Visualizer()
            annotated_image = visualizer.draw_detections(
                image_path, results[0], self.conf_threshold
            )

            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            saved_path = os.path.join(save_dir, f"{name}_detected{ext}")

            cv2.imwrite(saved_path, annotated_image)
            result_dict["saved_path"] = saved_path
            print(f"[检测] 结果已保存至: {saved_path}")

        # 显示结果
        if show_result and detections:
            from src.visualizer import Visualizer

            visualizer = Visualizer()
            annotated_image = visualizer.draw_detections(
                image_path, results[0], self.conf_threshold
            )
            visualizer.show_image(annotated_image, "YOLOv8 Detection Result")

        return result_dict

    def detect_video(
        self,
        video_path: str,
        save_path: Optional[str] = None,
        save_dir: str = "outputs/videos",
        show_result: bool = True,
        frame_skip: int = 1,
    ) -> Dict:
        """
        对视频文件执行目标检测

        Args:
            video_path: 输入视频路径
            save_path: 输出视频保存路径（None 则自动生成）
            save_dir: 输出视频保存目录
            show_result: 是否实时显示检测结果
            frame_skip: 跳帧数（1=每帧检测，2=隔帧检测）

        Returns:
            检测结果字典，包含:
            - video_path: 原始视频路径
            - total_frames: 总帧数
            - processed_frames: 实际处理帧数
            - total_detections: 检测到的目标总数
            - avg_inference_time_ms: 平均推理耗时(ms)
            - saved_path: 保存的视频路径
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        print(f"\n{'='*60}")
        print(f"[检测] 视频: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[检测] 视频信息: {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")

        # 准备输出视频写入器
        writer = None
        if save_path is None:
            os.makedirs(save_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(save_dir, f"{video_name}_detected.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps / frame_skip, (width, height))

        from src.visualizer import Visualizer

        visualizer = Visualizer()

        frame_count = 0
        processed_count = 0
        total_detections = 0
        total_inference_time = 0.0

        print(f"[检测] 开始处理视频... (按 'q' 键可提前退出)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧处理
            if frame_count % frame_skip != 0:
                continue

            processed_count += 1

            # 执行推理
            start_time = time.time()
            results = self.model(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            inference_time = (time.time() - start_time) * 1000
            total_inference_time += inference_time

            # 解析结果
            detections = self._parse_results(results)
            total_detections += len(detections)

            # 可视化
            annotated_frame = visualizer.draw_detections(
                frame, results[0], self.conf_threshold
            )

            # 添加 FPS 和帧数信息
            current_fps = 1000.0 / inference_time if inference_time > 0 else 0
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # 写入输出视频
            writer.write(annotated_frame)

            # 显示进度
            if processed_count % 10 == 0:
                progress = frame_count / total_frames * 100
                print(f"\r[进度] {progress:.1f}% ({frame_count}/{total_frames} 帧), "
                      f"平均推理时间: {total_inference_time/processed_count:.1f} ms", end="")

            # 实时显示
            if show_result:
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] 用户手动停止")
                    break

        # 清理资源
        cap.release()
        writer.release()
        if show_result:
            cv2.destroyAllWindows()

        avg_inference_time = total_inference_time / max(processed_count, 1)

        print(f"\n[检测] 视频处理完成!")
        print(f"[检测] 处理帧数: {processed_count}/{total_frames}")
        print(f"[检测] 检测目标总数: {total_detections}")
        print(f"[检测] 平均推理耗时: {avg_inference_time:.1f} ms")
        print(f"[检测] 结果视频已保存至: {save_path}")

        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "total_detections": total_detections,
            "avg_inference_time_ms": avg_inference_time,
            "saved_path": save_path,
        }

    def detect_batch(
        self,
        input_dir: str,
        save_dir: str = "outputs/images",
        show_result: bool = False,
    ) -> List[Dict]:
        """
        批量处理目录中的图像文件

        Args:
            input_dir: 输入图像目录
            save_dir: 结果保存目录
            show_result: 是否逐一显示结果

        Returns:
            每张图像的检测结果列表
        """
        from src.utils import is_image_file

        print(f"\n{'='*60}")
        print(f"[批量检测] 目录: {input_dir}")

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        image_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if is_image_file(f)
        ])

        print(f"[批量检测] 找到 {len(image_files)} 张图像")

        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
            result = self.detect_image(
                image_path=image_path,
                save_result=True,
                save_dir=save_dir,
                show_result=show_result,
            )
            results.append(result)

        # 汇总统计
        total_objects = sum(r["num_objects"] for r in results)
        total_time = sum(r["inference_time_ms"] for r in results)
        print(f"\n{'='*60}")
        print(f"[批量检测] 完成!")
        print(f"[批量检测] 处理图像: {len(results)} 张")
        print(f"[批量检测] 检测目标总数: {total_objects}")
        print(f"[批量检测] 总耗时: {total_time:.1f} ms")

        return results

    def detect_camera(
        self,
        camera_id: int = 0,
        save_video: bool = False,
        save_dir: str = "outputs/videos",
        show_result: bool = True,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        对摄像头实时视频流执行目标检测

        Args:
            camera_id: 摄像头 ID（0=默认摄像头）
            save_video: 是否保存检测结果视频
            save_dir: 输出视频保存目录
            show_result: 是否实时显示检测结果
            max_frames: 最大处理帧数（None=无限）

        Returns:
            检测结果字典
        """
        print(f"\n{'='*60}")
        print(f"[实时检测] 摄像头 ID: {camera_id}")
        print(f"[实时检测] 按 'q' 键退出 | 按 's' 键保存当前帧")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 ID={camera_id}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"[实时检测] 摄像头分辨率: {width}x{height}, 源 FPS: {fps_source:.1f}")

        writer = None
        if save_video:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "camera_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps_source, (width, height))

        from src.visualizer import Visualizer

        visualizer = Visualizer()

        frame_count = 0
        total_detections = 0
        total_inference_time = 0.0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            start_time = time.time()
            results = self.model(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            inference_time = (time.time() - start_time) * 1000
            total_inference_time += inference_time

            detections = self._parse_results(results)
            total_detections += len(detections)

            annotated_frame = visualizer.draw_detections(frame, results[0], self.conf_threshold)

            curr_time = time.time()
            fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            avg_time = total_inference_time / frame_count

            info_text = (
                f"FPS: {fps_display:.1f} | Frame: {frame_count} | "
                f"Avg: {avg_time:.1f}ms | Detections: {len(detections)}"
            )
            cv2.putText(
                annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            if writer:
                writer.write(annotated_frame)

            if show_result:
                cv2.imshow("YOLOv8 Realtime Detection", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[INFO] 用户手动停止")
                    break
                elif key == ord("s"):
                    snap_path = os.path.join(save_dir, f"snapshot_{frame_count}.jpg")
                    cv2.imwrite(snap_path, annotated_frame)
                    print(f"[INFO] 已保存快照: {snap_path}")

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        if writer:
            writer.release()
        if show_result:
            cv2.destroyAllWindows()

        avg_inference_time = total_inference_time / max(frame_count, 1)

        print(f"\n[实时检测] 完成!")
        print(f"[实时检测] 处理帧数: {frame_count}")
        print(f"[实时检测] 目标总数: {total_detections}")
        print(f"[实时检测] 平均推理耗时: {avg_inference_time:.1f} ms")
        if save_video:
            print(f"[实时检测] 视频已保存至: {save_path}")

        return {
            "camera_id": camera_id,
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "avg_inference_time_ms": avg_inference_time,
            "saved_path": save_path if save_video else None,
        }

    def _parse_results(self, results) -> List[Dict]:
        """
        解析 YOLO 推理结果为结构化数据

        Args:
            results: YOLO 推理原始结果

        Returns:
            检测结果列表，每个元素包含:
            - class_id: 类别 ID
            - class_name: 类别名称
            - confidence: 置信度
            - bbox: 边界框 [x1, y1, x2, y2]
            - bbox_norm: 归一化边界框 [cx, cy, w, h]
        """
        from src.utils import get_class_name

        detections = []

        if results is None or len(results) == 0:
            return detections

        # 获取第一个（也是唯一一个）结果
        result = results[0] if isinstance(results, list) else results

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            # 获取边界框坐标（像素坐标）
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            xywh = boxes.xywhn[i].cpu().numpy().tolist() if boxes.xywhn is not None else None

            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())

            detection = {
                "class_id": class_id,
                "class_name": get_class_name(class_id, lang="zh"),
                "class_name_en": get_class_name(class_id, lang="en"),
                "confidence": confidence,
                "bbox": [round(v, 1) for v in xyxy],  # [x1, y1, x2, y2]
                "bbox_norm": [round(v, 4) for v in xywh] if xywh else None,
            }
            detections.append(detection)

        # 按置信度降序排序
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return detections

    def summary(self) -> Dict:
        """返回检测器配置摘要"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "model_loaded": self.model is not None,
        }


class OptimizedDetector:
    """
    基于 OpenCV-DNN 的轻量化目标检测器

    使用 ONNX 格式模型，无需 PyTorch/ultralytics 运行时依赖。
    适用于边缘设备、嵌入式部署场景。

    功能:
    - 加载 ONNX 模型进行推理
    - 手动实现预处理、后处理、NMS
    - 支持图像、视频、摄像头检测
    """

    def __init__(
        self,
        model_path: str,
        names_path: str = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple = (640, 640),
    ):
        """
        初始化优化检测器

        Args:
            model_path: ONNX 模型文件路径
            names_path: 类别名称文件路径（coco.names）
            conf_threshold: 置信度阈值
            nms_threshold: NMS IoU 阈值
            input_size: 模型输入尺寸 (w, h)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        from src.utils import load_coco_names
        self.classes = load_coco_names(names_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        print(f"[INFO] 加载 ONNX 模型: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # 使用 OpenCV 后端（CPU），也可改为 DNN_BACKEND_CUDA 加速
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 获取输出层名称
        self.output_names = self.net.getUnconnectedOutLayersNames()
        print(f"[INFO] 模型输出层: {self.output_names}")
        print(f"[INFO] 输入尺寸: {input_size}, 类别数: {len(self.classes)}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：BLOB 转换 + 归一化 + 尺寸调整

        Args:
            image: 原始图像 (BGR)

        Returns:
            预处理后的 BLOB
        """
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / 255.0,
            self.input_size,
            swapRB=True,
            crop=False,
        )
        return blob

    def postprocess(
        self,
        outputs,
        image_shape: tuple,
    ) -> tuple:
        """
        YOLOv8 后处理：解析输出 + NMS

        YOLOv8 ONNX 输出形状: (1, 4 + num_classes, 8400)，每列为 [cx, cy, w, h, score_1, ..., score_80]

        Args:
            outputs: 模型原始输出
            image_shape: (height, width)

        Returns:
            (boxes, confidences, class_ids) 过滤后的检测结果
        """
        boxes = []
        confidences = []
        class_ids = []

        img_h, img_w = image_shape[:2]

        # YOLOv8 输出: (batch, 4+num_classes, num_anchors) -> 转置为 (batch, num_anchors, 4+num_classes)
        output = outputs[0] if isinstance(outputs, list) else outputs
        output = np.array(output).squeeze()  # 去掉 batch 维度

        # 确保形状：应是 (84, 8400) 或 (8400, 84)
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T  # 转置为 (8400, 84)

        # 计算缩放系数：模型输入 640x640 到原图尺寸
        x_scale = img_w / self.input_size[0]
        y_scale = img_h / self.input_size[1]

        num_classes = len(self.classes)

        for detection in output:
            # detection: [cx, cy, w, h, score_0, score_1, ..., score_79]
            bbox = detection[:4]
            scores = detection[4:]

            if len(scores) < num_classes:
                continue

            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > self.conf_threshold:
                cx, cy, w, h = bbox

                # 将中心点+宽高 转 左上+宽高，并按比例还原到原图坐标
                x = int((cx - w / 2) * x_scale)
                y = int((cy - h / 2) * y_scale)
                bw = int(w * x_scale)
                bh = int(h * y_scale)

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        # NMS 非极大值抑制
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.nms_threshold,
        )

        final_boxes, final_confidences, final_class_ids = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_confidences, final_class_ids

    def detect(self, image: np.ndarray) -> Dict:
        """
        检测单帧图像

        Args:
            image: 输入图像

        Returns:
            检测结果字典
        """
        blob = self.preprocess(image)
        self.net.setInput(blob)

        start_time = time.time()
        outputs = self.net.forward(self.output_names)
        inference_time = (time.time() - start_time) * 1000

        boxes, confidences, class_ids = self.postprocess(outputs, image.shape)

        detections = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            detections.append({
                "class_id": class_ids[i],
                "class_name": self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else f"class_{class_ids[i]}",
                "confidence": confidences[i],
                "bbox": [x, y, x + w, y + h],  # xyxy 格式
            })

        return {
            "detections": detections,
            "num_objects": len(detections),
            "inference_time_ms": inference_time,
        }

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        在图像上绘制检测框

        Args:
            image: 原始图像
            result: detect() 返回的结果字典

        Returns:
            标注后的图像
        """
        annotated = image.copy()
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
        return annotated

    def detect_image(self, image_path: str, save_result: bool = True,
                     save_dir: str = "outputs/images") -> Dict:
        """对图像文件执行检测"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        result = self.detect(image)
        annotated = self.visualize(image, result)

        print(f"\n[检测] 图像: {image_path}")
        print(f"[检测] 发现 {result['num_objects']} 个目标 | 耗时: {result['inference_time_ms']:.1f} ms")

        if save_result:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            saved_path = os.path.join(save_dir, f"{name}_onnx_detected{ext}")
            cv2.imwrite(saved_path, annotated)
            print(f"[检测] 结果已保存至: {saved_path}")
            result["saved_path"] = saved_path

        return result

    def detect_camera(self, camera_id: int = 0, show_result: bool = True) -> Dict:
        """对摄像头视频流执行检测（OpenCV-DNN 版本）"""
        print(f"\n[实时检测-ONNX] 摄像头 ID: {camera_id} | 按 'q' 键退出")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 ID={camera_id}")

        frame_count = 0
        total_time = 0.0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = self.detect(frame)
            annotated = self.visualize(frame, result)
            total_time += result["inference_time_ms"]

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            avg_time = total_time / frame_count
            info = (
                f"FPS: {fps:.1f} | Avg: {avg_time:.1f}ms | "
                f"Objects: {result['num_objects']} | [ONNX-DNN]"
            )
            cv2.putText(annotated, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if show_result:
                cv2.imshow("Optimized YOLOv8 Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if show_result:
            cv2.destroyAllWindows()

        avg_time = total_time / max(frame_count, 1)
        print(f"\n[实时检测-ONNX] 完成 | 处理 {frame_count} 帧 | 平均耗时: {avg_time:.1f} ms")
        return {"frames_processed": frame_count, "avg_inference_time_ms": avg_time}
