import torch.nn as nn
from transformer.module.MultiHeadAttention import MultiHeadAttention
from transformer.module.FeedForwardNetwork import FeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # Gap with DecoderLayer : Just a MHA.
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) # Gap with DecoderLayer, :norm3
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source_mask=None):
        # Gap with DecoderLayer
        # DecoderLayer
        # MHA > dropout > residual + Norm >
        # CrossAttention > dropout > residual + Norm >
        # FFN > dropout > residual + Norm
        residual = x
        x = self.self_attention(x, x, x, mask=source_mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)

        return x