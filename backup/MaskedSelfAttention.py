import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# 1. 데이터 준비
# -------------------------
batch_size = 1
seq_len = 6
d_k = 4

torch.manual_seed(0)

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)
                          # seq_len, d_k >>> d_k, seq_len
scores = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
                                # 6   ,   6
masked = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).bool() # [6,6]
scores = scores.masked_fill(masked, float('-inf'))

# scores와 브로드캐스트 호환을 위해 [1, seq_len, seq_len]로 확장

# -------------------------
# 4. Softmax → Attention Weights
# -------------------------
#weights = F.softmax(scores, dim=-1)
weights = F.softmax(scores, dim=-1)[0].detach().numpy()

plt.figure(figsize=(7,6))
sns.heatmap(weights, annot=True, cmap="Blues", cbar=True, square=True, xticklabels=[f"Token {i}" for i in range(seq_len)], yticklabels=[f"Token {i}" for i in range(seq_len)])
plt.title("Masked Self Attention Weights")
plt.xlabel("Key Positions (Attended To)")
plt.ylabel("Query Positions (Attending)")
plt.show()