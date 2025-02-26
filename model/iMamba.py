import torch
import torch.nn as nn
from layers.iMamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from mamba_ssm import Mamba
import math
__all__ = ['moving_avg', 'series_decomp']           

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.decomposition = configs.decomposition
        self.kernel_size = configs.kernel_size

        # 归一化配置
        if self.use_norm:
            self.norm = nn.LayerNorm(configs.d_model)

        # 序列分解模块
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            # 趋势项编码器
            self.enc_embedding_trend = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, 
                                                            configs.freq, configs.dropout)
            self.encoder_trend = Encoder(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1,
                        output_attention=self.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=4,  # Local convolution width
                        expand=2,  # Block expansion factor)
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
            self.projector_trend = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            
            # 周期项编码器
            self.enc_embedding_res = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed,
                                                           configs.freq, configs.dropout)
            self.encoder_res = Encoder(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1,
                        output_attention=self.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=4,  # Local convolution width
                        expand=2,  # Block expansion factor)
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
            self.projector_res = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        else:
            # 原始单分支结构
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed,
                                                      configs.freq, configs.dropout)
            self.encoder = Encoder(
                [EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=0.1,
                        output_attention=self.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=4,  # Local convolution width
                        expand=2,  # Block expansion factor)
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化处理
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, L, N = x_enc.shape

        if self.decomposition:
            # 序列分解
            res_init, trend_init = self.decomp_module(x_enc)
            
            # 趋势项处理
            enc_trend = self.enc_embedding_trend(trend_init, x_mark_enc)
            enc_trend, attns_trend = self.encoder_trend(enc_trend)
            dec_trend = self.projector_trend(enc_trend).permute(0, 2, 1)[:, :, :N]
            
            # 周期项处理
            enc_res = self.enc_embedding_res(res_init, x_mark_enc)
            enc_res, attns_res = self.encoder_res(enc_res)
            dec_res = self.projector_res(enc_res).permute(0, 2, 1)[:, :, :N]
            
            dec_out = dec_trend + dec_res
        else:
            # 原始处理流程
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out)
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # 反归一化
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
