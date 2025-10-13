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
        Lq = Q.size(1)
        Lk = K.size(1)
        Lv = V.size(1)
        # 의문,
        # 왜 self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)를 할까?
        # 왜 self.W_q(Q).view(batch_size, self.num_heads, -1, self.d_k)는 안되는걸까?
        # -1이 자동연산 영역이기때문이다.
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)  # -> (1,1,Lq,Lk)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # -> (B,1,Lq,Lk)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, h, Lq, d_k)

        # concat heads
        out = out.transpose(1, 2).contiguous().view(batch_size, Lq, self.d_model)
        return self.fc(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #return self.linear2(F.relu(self.linear1(x))) # Linear>ReLU>Linear
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) # Linear>dropout>ReLU>Linear

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, target_mask=None, enc_mask=None):
        # Masked
        _x = x  # _x:residual
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


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._generate_positional_encoding(max_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _generate_positional_encoding(selfself, max_len, d_model):
        # 이부분에 대한 설명이 좀 필요할거같다.
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(d_model).unsqueeze(0).float()
        angle = pos / torch.pow(10000, (2 * (i // 2)) / d_model)
        pe = torch.zeros(max_len, d_model)
        # 여기는 Attention의 기본 구조와 흡사하다.
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, target_tokens, enc_output, target_mask=None, enc_mask=None):
        # x의 두번째 차원 정수는 무엇을 의미하는가?
        #seq_len = x.size(1)
        B, T = target_tokens.size()
        x = self.embedding(target_tokens)  * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pos_encoding[:,-1,:].to(x.device)

        for layer in self.layers:
            x = layer(x, enc_output, target_mask=target_mask, enc_mask=enc_mask)

        logits = self.linear(x) # (B, T, vocab_size)
        return logits

"""
batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4
d_ff = 64

input = torch.randn(batch_size, seq_len, d_model)
enc_output = torch.randn(batch_size, seq_len, d_model)
decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0) #[1,1,5,5]
output = decoder_layer(input, enc_output, target_mask=mask)

#print("DecoderLayer output shape:", output.shape)

print("입력:", input.shape)
print("출력:", output.shape)
"""
# hyperparams for test
batch_size = 2
source_len = 7
tgt_len = 5
vocab_size = 100
d_model = 32
num_heads = 4
d_ff = 128
num_layers = 3

# dummy inputs
enc_output = torch.randn(batch_size, source_len, d_model)
target_tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))

# causal mask (1 = keep, 0 = mask)
causal = torch.tril(torch.ones(tgt_len, tgt_len)).bool()  # (T, T)
# expand to (B, T, T)
tgt_mask = causal.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, T)

decoder = Decoder(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                  num_heads=num_heads, d_ff=d_ff, max_len=50)

logits = decoder(target_tokens, enc_output, target_mask=tgt_mask)
print("Logits shape:", logits.shape)  # expected (B, T, vocab_size) [2,5,100]