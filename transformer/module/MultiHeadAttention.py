import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Lq = Q.size(1)
        Lk = K.size(1)
        Lv = V.size(1)

        Q = self.W_q(Q).view(B,Lq,self.num_heads,self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B,Lk,self.num_heads,self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B,Lv,self.num_heads,self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf")) # -1e9

        attention = F.softmax(scores, dim=-1)
        output = attention @ V
        output = output.transpose(1,2).contiguous().view(B,Lq,self.d_model)
        return self.W_o(output)