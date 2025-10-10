import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X):
        batch_size, seq_len, d_model = X.size()

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2,-1)/torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)))
        weigths = F.softmax(scores, dim=-1)
        Z = weigths @ V
        Z = Z.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_O(Z)

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, mha_out):
        residual = x + mha_out
        out = self.layernorm(residual)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,heads,d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, heads)
        self.resnorm1 = ResidualLayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
        self.resnorm2 = ResidualLayerNorm(d_model)

    def forward(self, x):
        mha_out = self.mha(x)
        x = self.resnorm1(x,mha_out)
        ffn_out = self.ffn(x)
        x = self.resnorm2(x,ffn_out)
        return x

batch_size = 2
seq_len = 5
d_model = 8*2
x = torch.randn(batch_size,seq_len,d_model)
encoder_layer = TransformerEncoderLayer(d_model=16,heads=4,d_ff=64)
out = encoder_layer(x)
print(f"Encoder Layer Output${out.shape}")