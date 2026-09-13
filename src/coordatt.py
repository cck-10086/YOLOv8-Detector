"""
坐标注意力模块 (Coordinate Attention, CA)

论文: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

特点:
  1. 在通道注意力的基础上嵌入位置信息, 分别沿水平/垂直方向进行池化
  2. 适用于小目标与低对比度缺陷场景 (如光伏EL图像中的裂纹、断栅)
  3. 参数量与计算量小, 不改变特征图尺寸与通道数, 可直接嵌入YOLO网络
"""

import torch
import torch.nn as nn


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


def register_custom_modules():
    """
    将自定义模块注册到 Ultralytics 框架, 使 YAML 配置文件可以引用 CoordAtt

    原理: ultralytics.nn.tasks.parse_model 通过模块名的全局查找表实例化模块,
          将 CoordAtt 注入其命名空间后, YAML 中即可直接书写 `CoordAtt`
    """
    import ultralytics.nn.tasks as tasks

    tasks.__dict__["CoordAtt"] = CoordAtt
    print("[INFO] 自定义模块 CoordAtt 已注册到 Ultralytics")
