# 모든 단계에 LayerNorm을 반복한다
# x_1 = LayerNorm(x + MaskedSelfAttention(x))
# x_2 = LayerNorm(x_1 + CrossAttention(x_1,EncOutput))
# x_3 = LayerNorm(x_2 + FFN(x_2))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Definitions
# MultiHeadAttention
# PositionwiseFeedForward
# DecoderLayer(MultiHeadAttention + PositionwiseFeedForward)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V는 각기 다른 SEED를 갖고, 결과가 같지않다.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc  = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 의문,
        # 왜 self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)를 할까?
        # 왜 self.W_q(Q).view(batch_size, self.num_heads, -1, self.d_k)는 안되는걸까?
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        Z = attn @ V
        Z = Z.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(Z)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x))) # Linear>ReLU>Linear

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, target_mask=None, enc_mask=None):
        # Masked
        _x = x  # TEMP
        x = self.self_attn(x, x, x, mask=target_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)

        # Encoder-Decoder
        _x = x
        x = self.cross_attn(x, enc_output, enc_output, mask=enc_mask)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        # FFN
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)

        return x

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4
d_ff = 32

decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

x = torch.randn(batch_size, seq_len, d_model)
enc_output = torch.randn(batch_size, seq_len, d_model)

mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0) #[1,1,5,5]
output = decoder_layer(x, enc_output, target_mask=mask)

print("DecoderLayer output shape:", output.shape)