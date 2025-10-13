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

        # B, L, D
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # (B, L, h, d_k) 형태로 변환
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # weights = F.softmax(scores, dim=-1)
        # Z = weights @ V
        # 2,8,2,5
        # 2,8,5,2
        # 5X2 @ 2X5 = 5X5
        scores = (Q @ K.transpose(-2,-1)/torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)))
        print(scores.shape) # 2,8,5,5
        weigths = F.softmax(scores, dim=-1)# 2,8,5,5
        Z = weigths @ V # 2,8,5,5 @ 2,8,5,2
        print(V.shape) #  2,8,5,2
        print(Z.shape) #  2,8,5,2

        Z = Z.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        # Z = (batch, head, seq, d_k) = (2, 8, 5, 2)
        # > (batch, seq, head, d_k) = (2, 5, 8, 2)
        # d_model = head*d_k
        # (batch, seq, head * d_k) = (2, 5, 16)
        return self.W_O(Z)

X = torch.rand(2,5,16)
mha = MultiHeadAttention(d_model=16, num_heads=8)
out = mha(X)

#print(out.shape)