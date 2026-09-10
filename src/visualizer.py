"""
可视化模块 - 检测结果的可视化展示
包括边界框绘制、类别标签显示和置信度标注
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


# 预定义颜色列表 (BGR 格式)，用于不同类别
COLORS = [
    (56, 56, 255),   # 红色
    (0, 255, 127),   # 春绿色
    (255, 218, 100), # 橙色
    (219, 112, 147), # 粉色
    (52, 211, 246),  # 黄色
    (230, 186, 64),  # 青色
    (0, 128, 255),   # 橙红
    (226, 43, 138),  # 紫红
    (187, 119, 53),  # 棕色
    (128, 0, 128),   # 紫色
    (0, 255, 0),     # 绿色
    (255, 0, 0),     # 蓝色
    (152, 251, 152), # 浅绿
    (203, 192, 255), # 浅粉
    (220, 245, 220), # 淡黄
    (0, 255, 255),   # 黄色
    (255, 255, 0),   # 天蓝
    (255, 0, 255),   # 品红
]


class Visualizer:
    """
    检测结果可视化器

    功能:
    - 在图像上绘制边界框
    - 标注类别标签和置信度
    - 支持自定义颜色和样式
    """

    def __init__(
        self,
        box_thickness: int = 2,
        font_size: float = 0.5,
        font_thickness: int = 2,
        show_confidence: bool = True,
    ):
        """
        初始化可视化器

        Args:
            box_thickness: 边界框线条粗细
            font_size: 字体大小比例
            font_thickness: 字体粗细
            show_confidence: 是否显示置信度
        """
        self.box_thickness = box_thickness
        self.font_size = font_size
        self.font_thickness = font_thickness
        self.show_confidence = show_confidence

    def draw_detections(
        self,
        image: np.ndarray,
        yolo_result,
        conf_threshold: float = 0.0,
    ) -> np.ndarray:
        """
        在图像上绘制所有检测结果

        Args:
            image: 输入图像 (numpy array, BGR 格式)
            yolo_result: YOLO 推理结果对象
            conf_threshold: 置信度阈值（仅显示高于此值的结果）

        Returns:
            带标注的图像
        """
        from src.utils import get_class_name

        # 如果输入是文件路径，则加载图像
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {image}")

        # 复制图像，避免修改原图
        annotated = image.copy()

        if yolo_result is None or yolo_result.boxes is None:
            return annotated

        boxes = yolo_result.boxes

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].item())

            # 过滤低置信度检测
            if confidence < conf_threshold:
                continue

            class_id = int(boxes.cls[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy.tolist()

            # 获取颜色
            color = COLORS[class_id % len(COLORS)]

            # 绘制边界框
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                self.box_thickness,
                cv2.LINE_AA,
            )

            # 准备标签文本
            class_name = get_class_name(class_id, lang="zh")
            if self.show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name

            # 绘制标签
            annotated = self._draw_label(annotated, label, x1, y1, color)

        return annotated

    def _draw_label(
        self,
        image: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        绘制带背景的文本标签

        Args:
            image: 图像
            label: 标签文本
            x: 左上角 x 坐标
            y: 左上角 y 坐标
            color: 边框颜色 (B, G, R)

        Returns:
            标注后的图像
        """
        # 计算文本尺寸
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thickness
        )

        # 标签背景位置
        label_y = max(y - text_h - 8, 0)
        label_x1 = x
        label_x2 = x + text_w + 6
        label_y1 = label_y
        label_y2 = y

        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,  # 填充
        )
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # 绘制标签文本（白色文字）
        cv2.putText(
            image,
            label,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_size,
            (255, 255, 255),  # 白色文字
            self.font_thickness,
            cv2.LINE_AA,
        )

        return image

    def show_image(
        self,
        image: np.ndarray,
        window_name: str = "Detection Result",
        wait_key: bool = True,
    ) -> None:
        """
        显示图像

        Args:
            image: 图像数据
            window_name: 窗口名称
            wait_key: 是否等待按键关闭
        """
        cv2.imshow(window_name, image)
        if wait_key:
            print("[INFO] 按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def create_comparison(
        self,
        original_image: np.ndarray,
        detected_image: np.ndarray,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        创建原图与检测结果的并排对比图

        Args:
            original_image: 原始图像
            detected_image: 检测结果图像
            save_path: 保存路径（可选）

        Returns:
            并排对比图像
        """
        # 确保两张图高度一致
        h = max(original_image.shape[0], detected_image.shape[0])

        if original_image.shape[0] < h:
            original_image = cv2.resize(
                original_image,
                (int(original_image.shape[1] * h / original_image.shape[0]), h),
            )
        if detected_image.shape[0] < h:
            detected_image = cv2.resize(
                detected_image,
                (int(detected_image.shape[1] * h / detected_image.shape[0]), h),
            )

        # 添加标题
        cv2.putText(
            original_image, "Original",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.putText(
            detected_image, "Detected",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        comparison = np.hstack([original_image, detected_image])

        if save_path:
            cv2.imwrite(save_path, comparison)

        return comparison

    def print_detections_table(self, detections: List[Dict]) -> None:
        """
        以表格形式打印检测结果

        Args:
            detections: 检测结果列表
        """
        if not detections:
            print("[INFO] 未检测到任何目标")
            return

        print(f"\n{'='*65}")
        print(f" 检测结果汇总")
        print(f"{'='*65}")
        print(f" {'ID':<4} {'类别':<30} {'置信度':<10} {'边界框 (x1,y1,x2,y2)':<30}")
        print(f" {'-'*63}")

        for i, det in enumerate(detections):
            class_name = det.get("class_name", "未知")
            confidence = det.get("confidence", 0)
            bbox = det.get("bbox", [0, 0, 0, 0])
            bbox_str = f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})"

            print(f" {i+1:<4} {class_name:<30} {confidence:<10.3f} {bbox_str:<30}")

        print(f" {'='*63}")
        print(f" 共检测到 {len(detections)} 个目标")
        print(f" {'='*63}\n")
