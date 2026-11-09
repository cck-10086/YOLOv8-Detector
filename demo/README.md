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
| 权重 | `best.pt`（6.0 MB，论文最终方案 `ca-deep-real2`）；创空间仓库里存为 `best.weights` |
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

## 在线 Demo（魔搭创空间）

已部署在魔搭 ModelScope 创空间，国内直连、免费 CPU，无需信用卡：

https://modelscope.cn/studios/cck10086/pv-defect-detector

## 自己部署一份

推荐用**魔搭 ModelScope 创空间**（免费 CPU、原生 Gradio、国内访问快）。
HuggingFace Spaces 的 Gradio/Docker 空间现在需要 PRO 付费（$9/月并绑卡），
Static 空间免费但跑不了 Python，所以本项目不走 HF。

本目录已经按创空间的要求准备好：`README.md` 顶部有 HF 格式元信息（创空间兼容）、
`app.py` 是入口、`requirements.txt` 声明依赖、`best.pt` 是权重、`src/` 是自定义模块。

1. 注册 modelscope.cn，进「创空间」→ 新建，SDK 选 Gradio，硬件选**免费 CPU**。
2. 把本目录**全部文件**上传上去（含权重和 `src/`）。注意 `app.py` 要放在
   创空间根目录，不是 `demo/` 子目录。
3. 部署设置里镜像选 py312 + torch 的官方镜像（自带 torch/torchvision，
   所以 `requirements.txt` 只装 ultralytics、gradio、pandas）。
4. 点「确认并部署」，等约 10 分钟构建完成。

三个坑已经在本目录的代码里处理好，不用再改：

- **上传会拍平子目录**：`src/coordatt.py` 会落到创空间根目录，但权重的
  pickle 里写死了 `src.coordatt.C2f_CA`。`app.py` 的 `_ensure_src_package()`
  会注册一个指向当前目录的命名空间包 `src`，让反序列化能成功。
- **权重不要用 `.pt` 后缀上传**：创空间仓库的 `.gitattributes` 把 `*.pt`
  交给 git-lfs，而 LFS 下载链路会报 `Mirror forwarding loop detected`，
  构建时 `git clone` 直接失败。所以仓库里存成 `best.weights`（普通 git 对象）。
  但 ultralytics 只认 `.pt` 后缀，`app.py` 的 `_as_pt_path()` 会在加载前
  把它复制成临时目录下的 `.pt` 再用。
- **必须监听 0.0.0.0:7860**：`app.py` 结尾用 `GRADIO_SERVER_NAME` /
  `GRADIO_SERVER_PORT` 环境变量控制，默认就是创空间要求的地址和端口。

本地跑不需要这些设置，`python demo/app.py` 直接起在 http://127.0.0.1:7860 。
