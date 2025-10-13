import torch
import torch.nn.functional as F

X = torch.tensor([
    [1.0,0.5,0.3,0.2],
    [0.1,0.8,0.4,0.7],
    [0.6,0.9,0.2,0.3]
])


d_model = X.size(1) # 2차원, 행렬의 '열' 개수, 0는 3으로 나옴
d_k = d_model

W_Q = torch.rand(d_model,d_k)   # 4,4
W_K = torch.rand(d_model,d_k)
W_V = torch.rand(d_model,d_k)

# 난수생성의 원칙대로면, 셋은 같은 SEED를 가지고만들었다.
Q = X @ W_Q # 3X4 의 랜덤행렬
K = X @ W_K
V = X @ W_V

# 점수를 왜 Query에서만 확인하는가?
# T : Transpose Default (A,B) > (B,A)로 변환
scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))


weights = F.softmax(scores, dim=-1)

Z = weights @ V

print(f"Attention Socres\n${scores}")
print("")
print(f"Attention Weigths (Softmax)\n${weights}")
print("")
print(f"Attention Output\n${Z}")