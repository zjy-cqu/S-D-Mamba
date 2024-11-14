import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class EncoderLayer(nn.Module):
    def __init__(self, mamba, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.mamba = mamba  # 使用 Mamba 模块
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, tau=None, delta=None):
        # 使用 Mamba 模块
        x = self.norm1(x)
        mamba_out = self.mamba(x)  # 用 Mamba 处理输入
        x = x + self.dropout(mamba_out)  # 残差连接

        x = self.norm2(x)
        return x, None  # 不使用 attention，所以返回 None


class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)  # 使用 Mamba 层
        self.norm = norm_layer

    def forward(self, x, tau=None, delta=None):
        # x: [B, L, D]
        for mamba_layer in self.mamba_layers:
            x, _ = mamba_layer(x, tau=tau, delta=delta)  # Mamba 层不使用 attention

        if self.norm is not None:
            x = self.norm(x)

        return x, None  # 不返回 attention，所以返回 None
