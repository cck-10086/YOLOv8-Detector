"""
光伏电池片缺陷检测 Demo

模型：YOLOv8n + 深层坐标注意力（第 6、8 层注入），论文最终方案
数据：PVEL-AD 光伏电池片电致发光图像，8 类主要缺陷
用途：上传一张电池片图像，返回带检测框的结果图和缺陷统计

本地运行：
    python demo/app.py
"""

import sys
from pathlib import Path

import gradio as gr
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

# 权重是带自定义模块的 checkpoint，反序列化时要能 import 到 src.coordatt。
# 本地跑时 src/ 在仓库根（ROOT 的上一级），部署到 HuggingFace Spaces 时
# app.py 就在根目录、src/ 与它同级，所以两个位置都加进 sys.path。
for _p in (str(PROJECT_ROOT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 类别中文名，用于界面展示；顺序与模型 names 一致
CLASS_CN = {
    "finger": "指状缺陷",
    "crack": "裂纹",
    "black_core": "黑芯",
    "thick_line": "粗线",
    "horizontal_dislocation": "水平错位",
    "vertical_dislocation": "垂直错位",
    "short_circuit": "短路",
    "star_crack": "星状裂纹",
}

ROOT = Path(__file__).resolve().parent
WEIGHT = ROOT / "best.pt"

# 模型只加载一次，避免每次推理都读盘
_model = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHT.exists():
            raise FileNotFoundError(f"权重文件不存在：{WEIGHT}")
        _model = YOLO(str(WEIGHT))
    return _model


def detect(image, conf: float, iou: float):
    """对单张图像做推理，返回（结果图, 统计表, 文字摘要）。"""
    if image is None:
        return None, None, "请先上传一张光伏电池片图像。"

    model = get_model()
    results = model.predict(
        source=image,
        conf=float(conf),
        iou=float(iou),
        verbose=False,
    )
    result = results[0]

    # result.plot() 返回 BGR 数组，Gradio 需要 RGB
    annotated = result.plot()[:, :, ::-1]

    names = result.names
    counts = {}
    confs = {}
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id, score in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            label = names[int(cls_id)]
            counts[label] = counts.get(label, 0) + 1
            confs.setdefault(label, []).append(float(score))

    if not counts:
        return annotated, pd.DataFrame(
            columns=["缺陷类型", "数量", "平均置信度"]
        ), "未检测到缺陷。可以调低置信度阈值再试一次。"

    rows = []
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        avg = sum(confs[label]) / len(confs[label])
        rows.append(
            {
                "缺陷类型": f"{CLASS_CN.get(label, label)}（{label}）",
                "数量": n,
                "平均置信度": round(avg, 3),
            }
        )
    table = pd.DataFrame(rows)

    total = sum(counts.values())
    kinds = len(counts)
    summary = f"共检出 {total} 个缺陷，涉及 {kinds} 类。"
    top = table.iloc[0]
    summary += f" 最多的是 {top['缺陷类型']}，{top['数量']} 个。"
    return annotated, table, summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="光伏电池片缺陷检测") as demo:
        gr.Markdown(
            """
            # 光伏电池片缺陷检测

            YOLOv8n + 深层坐标注意力（第 6、8 层注入），在 PVEL-AD 数据集上
            mAP50 达到 **90.17%**，比 YOLOv8n 基线提升约 2.1 个百分点。

            上传一张电池片电致发光（EL）图像，模型会框出缺陷位置并统计类别。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="输入图像", type="numpy")
                conf = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.05,
                    label="置信度阈值",
                    info="调低会检出更多候选框，误检也会增加",
                )
                iou = gr.Slider(
                    0.1, 0.9, value=0.45, step=0.05,
                    label="NMS IoU 阈值",
                    info="两个框重叠超过该比例时只保留置信度高的那个",
                )
                btn = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(label="检测结果", type="numpy")
                summary = gr.Textbox(label="摘要", lines=2)
                table = gr.Dataframe(label="缺陷统计", interactive=False)

        btn.click(detect, [image_in, conf, iou], [image_out, table, summary])
        image_in.change(detect, [image_in, conf, iou], [image_out, table, summary])

        gr.Markdown(
            """
            ### 已知限制

            - 训练数据只有 8 类主要缺陷，另 4 类极稀缺缺陷（printing_error、
              corner、fragment、scratch）样本太少，模型基本检不出。
            - 召回率 82.04%，比基线低 2 个百分点：模型更准但更保守，小目标
              容易漏。这是刻意的取舍，不是调参失误。
            - 训练只用了单一种子，没有跑多次取方差，指标存在随机波动。
            """
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
