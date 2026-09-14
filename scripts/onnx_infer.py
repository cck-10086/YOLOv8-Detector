# -*- coding: utf-8 -*-
"""
ONNX 独立部署推理演示 (不依赖 PyTorch / Ultralytics)

仅使用 onnxruntime + numpy + opencv + pillow:
  letterbox 预处理 -> onnxruntime 推理 -> 置信度过滤 -> NMS -> 中文标签可视化
模拟真实产线部署环境, 验证模型可脱离训练框架独立运行。
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# 类别英文 -> 中文名 (界面显示使用中文)
CLASS_ZH = {
    0: "断栅", 1: "裂纹", 2: "黑芯", 3: "增厚栅线",
    4: "横向错位", 5: "短路", 6: "纵向错位", 7: "星形裂纹",
}
# 每类一个固定颜色 (BGR)
CLASS_COLOR = {
    0: (180, 105, 255), 1: (0, 0, 255), 2: (0, 0, 139), 3: (0, 165, 255),
    4: (255, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 7: (0, 255, 255),
}


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """保持长宽比缩放并填充到正方形, 返回预处理图与还原参数"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    padded = cv2.copyMakeBorder(resized, top, new_shape - nh - top,
                                left, new_shape - nw - left,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def nms(boxes, scores, iou_thr=0.45):
    """纯 numpy 单类 NMS, boxes 为 xyxy 格式"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(output, conf_thr=0.25):
    """ONNX 输出 (1, 4+nc, N) -> 检测框列表 [x1,y1,x2,y2,conf,cls] (640 尺度)"""
    pred = output[0].T  # (N, 4+nc)
    boxes_xywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)
    mask = confs > conf_thr
    boxes_xywh, confs, cls_ids = boxes_xywh[mask], confs[mask], cls_ids[mask]

    # xywh -> xyxy
    boxes = np.stack([boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
                      boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
                      boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2,
                      boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2], axis=1)

    final = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms(boxes[idx], confs[idx])
        for k in keep:
            i = idx[k]
            final.append([*boxes[i], confs[i], cls_ids[i]])
    return np.array(final) if final else np.empty((0, 6))


def draw(img_bgr, dets, scale, pad_x, pad_y, font):
    """将 640 尺度检测框还原到原图坐标并绘制 (中文标签)"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw_obj = ImageDraw.Draw(img_pil)
    for x1, y1, x2, y2, conf, cls in dets:
        x1 = max(0, (x1 - pad_x) / scale)
        y1 = max(0, (y1 - pad_y) / scale)
        x2 = min(img_pil.width, (x2 - pad_x) / scale)
        y2 = min(img_pil.height, (y2 - pad_y) / scale)
        cls = int(cls)
        color = CLASS_COLOR[cls][::-1]  # BGR -> RGB
        draw_obj.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{CLASS_ZH[cls]} {conf:.2f}"
        tb = draw_obj.textbbox((x1, y1), text, font=font)
        draw_obj.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw_obj.text((x1, y1), text, fill="white", font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def pick_sample_image(val_dir):
    """挑选一张含 3 个以上缺陷实例的验证集图片 (多目标展示效果更好)"""
    label_dir = val_dir.replace("images", "labels")
    for lf in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        if sum(1 for _ in open(lf, encoding="utf-8")) >= 3:
            stem = os.path.splitext(os.path.basename(lf))[0]
            for ext in (".jpg", ".png", ".JPG", ".PNG"):
                p = os.path.join(val_dir, stem + ext)
                if os.path.exists(p):
                    return p
    return sorted(glob.glob(os.path.join(val_dir, "*.*")))[0]


def main():
    parser = argparse.ArgumentParser(description="ONNX 独立部署推理演示")
    parser.add_argument("--onnx", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "runs/train/yolov8n-ca22/weights/best.onnx"))
    parser.add_argument("--image", type=str, default=None, help="待推理图片 (默认自动挑选验证集多目标图)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = args.image or pick_sample_image(
        os.path.join(project_root, "data/PVEL-AD/images/val"))

    # ---------- 推理准备 ----------
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 兼容中文路径
    blob, scale, pad_x, pad_y = letterbox(img_bgr, 640)
    x = blob[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW

    # ---------- 单张推理 ----------
    output = session.run(None, {input_name: x})[0]
    dets = postprocess(output, args.conf)

    # ---------- 测速 (预热后循环 20 次) ----------
    for _ in range(5):
        session.run(None, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(20):
        session.run(None, {input_name: x})
    avg_ms = (time.perf_counter() - t0) / 20 * 1000

    # ---------- 可视化 ----------
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    result = draw(img_bgr, dets, scale, pad_x, pad_y, font)
    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "onnx_deploy_demo.jpg")
    cv2.imencode(".jpg", result)[1].tofile(out_path)  # 兼容中文路径

    # ---------- 汇报 ----------
    print("==================== ONNX 独立部署推理 ====================")
    print(f"运行时: onnxruntime {ort.__version__} (CPUExecutionProvider, 无 PyTorch 依赖)")
    print(f"输入尺寸: 1x3x640x640")
    print(f"推理耗时: {avg_ms:.1f} ms/张")
    print(f"检出目标: {len(dets)} 个")
    for x1, y1, x2, y2, conf, cls in sorted(dets.tolist(), key=lambda d: -d[4]):
        print(f"  {CLASS_ZH[int(cls)]}  置信度 {conf:.3f}  位置({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    print(f"结果图: {out_path}")


if __name__ == "__main__":
    main()
