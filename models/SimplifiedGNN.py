# models/SimplifiedGNN.py
import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class SimplifiedGNN(nn.Module):
    """
    一个“简化版”的 GNN 层，实现方式与 LightGCN 一致的线性邻居聚合，
    但额外引入一个可学习的全局标量 gate（alpha），以及可选的残差项。
    目的：与 LGConv 接口一致 (x, edge_index) -> x'，可直接替换。

    参数（从 config 读取，均可缺省）:
      - simplified_residual: bool，是否开启残差（默认 False）
      - simplified_alpha_init: float，alpha 的初始化值（默认 1.0）
    """
    def __init__(self, config, edge_index=None):
        super().__init__()
        # 底层依旧用 LGConv 做规范化邻居聚合（等价于 LightGCN 单层）
        self.conv = LGConv(normalize=True)

        # 一个可学习的全局标量 gate，控制聚合强度；设为 1.0 等价于原始 LightGCN
        alpha_init = float(config.get('simplified_alpha_init', 1.0))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # 可选残差
        self.use_residual = bool(config.get('simplified_residual', False))
        if self.use_residual:
            # 残差系数也设为可学习标量
            self.res_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, edge_index):
        """
        与 LGConv 一致的签名：输入整图节点表征 x (N, d)，边 edge_index (2, E)，输出更新后的 x'。
        """
        out = self.conv(x, edge_index)         # 邻居聚合（已做度归一化）
        out = self.alpha * out                 # 全局门控
        if self.use_residual:
            out = out + self.res_scale * x     # 可选残差
        return out
