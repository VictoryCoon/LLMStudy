import torch.nn as nn
from transformer.module.MultiHeadAttention import MultiHeadAttention
from transformer.module.FeedForwardNetwork import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads) # Gap with EncoderLayer
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) # Gap with EncoderLayer, :norm2
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, target_mask=None, enc_mask=None):
        # Gap with EncoderLayer
        # EncoderLayer
        # MHA > dropout > residual + Norm >
        # FFN > dropout > residual + Norm
        residual = x
        x = self.self_attention(x, x, x, mask=target_mask)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        residual = x
        x = self.cross_attention(x, enc_output, enc_output, mask=enc_mask)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(residual + x)
        return x