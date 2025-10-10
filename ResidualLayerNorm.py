import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, mha_out):
        residual = x + mha_out
        out = self.layernorm(residual)
        return out

batch_size = 2
seq_len = 5
d_model = 16

x       = torch.randn(batch_size, seq_len, d_model) # 랜덤 입력
mha_out = torch.randn(batch_size, seq_len, d_model) # 랜덤...입력?(concat + W_O)

layer = ResidualLayerNorm(d_model)
out = layer(x,mha_out)

print(f"Input\n${x.shape}")
print("")
print(f"MHA Output\n${mha_out.shape}")
print("")
print(f"Residual + LayerNorm Output\n${out.shape}")