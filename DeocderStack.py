import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DecoderLayer 재사용 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        Z = attn @ V
        Z = Z.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(Z)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


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
        _x = x
        x = self.self_attn(x, x, x, mask=target_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.cross_attn(x, enc_output, enc_output, mask=enc_mask)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._generate_positional_encoding(max_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(selfself, max_len, d_model):
        # 이부분에 대한 설명이 좀 필요할거같다.
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)
        angle = pos / torch.pow(10000,(2*(i//2))/d_model)
        pe = torch.zeros(max_len, d_model)
        # 여기는 Attention의 기본 구조와 흡사하다.
        pe[:,0::2] = torch.sin(angle[:,0::2])
        pe[:,1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(0)
    
    def forward(self, x, enc_output, target_mask=None, enc_mask=None):
        # x의 두번째 차원 정수는 무엇을 의미하는가?
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:,:seq_len,:].to(x.device)

        for layer in self.layers:
            x = layer(x,enc_output,target_mask,enc_mask)

        logits = self.linear(x)
        return logits

# 실행
batch_size = 2
seq_len = 5
vocab_size = 100
d_model = 16
num_heads = 4
d_ff = 32
num_layers = 3

decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff)

# 더미 입력
x = torch.randint(0, vocab_size, (batch_size, seq_len))
enc_output = torch.randn(batch_size, seq_len, d_model)

mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

output = decoder(x, enc_output, target_mask=mask)
print("Decoder output shape:", output.shape)