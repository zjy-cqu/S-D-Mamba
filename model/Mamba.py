import torch
import torch.nn as nn
from layers.Mamba_pure_EncDec import Encoder, EncoderLayer  # 使用修改后的 Encoder 和 EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding layer
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, 
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy

        # Encoder-only architecture with Mamba only
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(  # 使用 Mamba 进行时序建模
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)  # 堆叠多个 EncoderLayer
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)  # 如果需要，加入 norm 层
        )

        # Output projector
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Optionally apply normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Pass through the encoder to get the final representation
        enc_out, _ = self.encoder(enc_out)  # 使用 Mamba 层进行时序建模

        # Project to the desired output shape and dimensions
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # Apply de-normalization if used
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
