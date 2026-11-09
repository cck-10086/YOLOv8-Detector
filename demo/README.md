---
title: 光伏电池片缺陷检测
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# 光伏电池片缺陷检测 Demo

YOLOv8n + 深层坐标注意力（第 6、8 层注入），PVEL-AD 数据集 mAP50 **90.17%**，
比 YOLOv8n 基线提升 2.13 个百分点。

上传一张电池片电致发光（EL）图像，模型框出缺陷位置并按类别统计。

## 模型说明

| 项 | 值 |
| --- | --- |
| 检测框架 | YOLOv8n + CoordAtt（注入 backbone 第 6、8 层） |
| 权重 | `best.pt`（6.0 MB，论文最终方案 `ca-deep-real2`） |
| 类别 | 8 类：指状缺陷、裂纹、黑芯、粗线、水平错位、垂直错位、短路、星状裂纹 |
| mAP50 | 90.17%（基线 88.04%） |
| 召回率 | 82.04%（基线 84.17%，**更准但更保守，这是有意取舍**） |

## 本地运行

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

打开 http://127.0.0.1:7860 。

## 已知限制

1. 训练数据只有 8 类主要缺陷，另 4 类极稀缺缺陷（printing_error、corner、
   fragment、scratch）样本太少，模型基本检不出。
2. 召回率比基线低 2 个百分点：小目标容易漏，换来的是一批误检被压掉。
3. 训练只用了单一种子，没有跑多次取方差，指标存在随机波动。
4. CPU 推理约 1–2 秒/张，GPU 更快。

## 部署到 HuggingFace Spaces

本目录已经按 Spaces 的格式准备好（`README.md` 顶部有元信息、`app.py` 是入口、
`requirements.txt` 声明依赖、`best.pt` 是权重、`src/` 是自定义模块）。

1. 注册 huggingface.co 账号，点 New Space，SDK 选 Gradio，硬件选 CPU（免费）。
2. 创建后进 Files 页，把本目录**全部文件**上传上去（含 `best.pt` 和 `src/`）。
   注意 `app.py` 要放在 Space 的根目录，不是 `demo/` 子目录。
3. 等 Building 结束，即可通过 `https://huggingface.co/spaces/<用户名>/<空间名>` 访问。

命令行方式（可选）：

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <用户名>/<空间名> demo --repo-type space
```

`app.py` 里对 `src` 的 import 路径同时兼容「仓库内 `demo/` 子目录」和
「Spaces 根目录」两种布局，不用改代码。
