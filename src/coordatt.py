"""
坐标注意力模块 (Coordinate Attention, CA)

论文: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

特点:
  1. 在通道注意力的基础上嵌入位置信息, 分别沿水平/垂直方向进行池化
  2. 适用于小目标与低对比度缺陷场景 (如光伏EL图像中的裂纹、断栅)
  3. 参数量与计算量小, 不改变特征图尺寸与通道数, 可直接嵌入YOLO网络

改进方案: 采用"继承式"设计 (C2f_CA 继承原生 C2f), 通过 inject_coord_attention()
在模型构建后原地替换 backbone 指定层的 C2f —— 参数名与 C2f 完全一致, COCO 预训练
权重可全量迁移 (仅新增 ca 分支从零学习), 且不依赖对 Ultralytics 库的任何修改。
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f


class CoordAtt(nn.Module):
    """坐标注意力模块

    Args:
        c1: 输入通道数
        reduction: 通道缩减比例 (默认: 32, 中间通道数不低于8)
    """

    def __init__(self, c1, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度方向池化 -> (N,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高度方向池化 -> (N,C,1,W)

        # 中间通道数, 不低于8防止信息瓶颈
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """前向传播: 生成方向感知的注意力权重并对输入加权"""
        identity = x
        n, c, h, w = x.size()

        # 1. 方向池化并拼接
        x_h = self.pool_h(x)                          # (N,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # (N,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)              # (N,C,H+W,1)

        # 2. 共享1x1卷积学习跨通道交互
        y = self.act(self.bn1(self.conv1(y)))

        # 3. 拆分并还原为两组方向注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()              # (N,C,H,1)
        a_w = self.conv_w(x_w).sigmoid()              # (N,C,1,W)

        # 4. 对输入特征加权 (不改变尺寸与通道数)
        return identity * a_w * a_h


class C2f_CA(C2f):
    """C2f 坐标注意力改进模块

    继承原生 C2f, 仅在 forward 中对每个 Bottleneck 输出施加坐标注意力。
    cv1/cv2/m 的参数名与形状和 C2f 完全一致, 预训练权重可全量迁移;
    新增的 ca 分支参数量极小, 从零学习。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.ca = CoordAtt(self.c)  # 新增的注意力分支 (从零学习)

    def forward(self, x):
        """每个 Bottleneck 输出经 CA 方向感知加权后再拼接"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(self.ca(m(y[-1])) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def inject_coord_attention(model_nn, stages=(2, 4, 6, 8)):
    """将 YOLO 模型 backbone 指定层的 C2f 原地替换为 C2f_CA, 并迁移全部原有权重

    Args:
        model_nn: ultralytics DetectionModel 实例 (model.model)
        stages: backbone 中 C2f 所在的层索引 (yolov8n 默认: 2/4/6/8)

    Returns:
        注入信息字符串 (各层通道与新增参数量)
    """
    replaced = []
    for idx in stages:
        old = model_nn.model[idx]
        if not isinstance(old, C2f):
            raise TypeError(f"第 {idx} 层是 {type(old).__name__}, 不是 C2f, 无法注入 CA")
        c1 = old.cv1.conv.in_channels
        c2 = old.cv2.conv.out_channels
        n = len(old.m)
        shortcut = bool(old.m[0].add) if n > 0 else False
        e = old.c / c2  # C2f 内部隐藏通道比例

        new = C2f_CA(c1, c2, n=n, shortcut=shortcut, e=e)
        # 参数名一致 (cv1/cv2/m), ca 分支为新增参数 -> strict=False 迁移全部旧权重
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        assert not unexpected, f"意外的多余参数: {unexpected}"
        assert all(k.startswith("ca.") for k in missing), f"存在非 ca 分支的缺失参数: {missing}"
        # 保留 Ultralytics 前向调度所需属性 (层索引/输入来源/类型标记)
        new.i, new.f, new.type = old.i, old.f, old.type
        model_nn.model[idx] = new
        replaced.append(f"layer{idx}(C{c2})")

    ca_params = sum(p.numel() for m in model_nn.modules() if isinstance(m, CoordAtt)
                    for p in m.parameters())
    total = sum(p.numel() for p in model_nn.parameters())
    return (f"已注入坐标注意力: {', '.join(replaced)} | "
            f"CA新增参数 {ca_params / 1e3:.1f}K, 模型总参数 {total / 1e6:.2f}M")


def register_custom_modules():
    """将自定义模块注册到 Ultralytics 命名空间 (供插入式 YAML 方案 yolov8n-ca.yaml 解析使用;
    继承式注入方案不依赖此注册)"""
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["CoordAtt"] = CoordAtt
    tasks.__dict__["C2f_CA"] = C2f_CA
    print("[INFO] 自定义模块 CoordAtt / C2f_CA 已注册到 Ultralytics")
