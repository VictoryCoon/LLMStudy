import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads # 4
        self.d_k = d_model // num_heads # 16 // 4 = 4
        self.d_model = d_model # 16

        self.W_q = nn.Linear(d_model, d_model) # 16X16
        self.W_k = nn.Linear(d_model, d_model) # 16X16
        self.W_v = nn.Linear(d_model, d_model) # 16X16
        self.W_o = nn.Linear(d_model, d_model) # 16X16

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # [2,5,16] > Q.size(0) = [2,-1,4,4] > [2,4,(Auto-Calc),4](*2X5X16의 인수분해구나🙃) > [2,4,5,5]
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # [2,4,5,4]@[2,4,4,5] / sqrt(4) = 2 = [2,4,5,5]
        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # Softmax는 인자값 x와 인자값에대한 인덱스i를 받는다
        # Softmax(x_i) = F.softmax(scores=x, dim=i)
        weights = F.softmax(scores, dim=-1) # softmax는 차원합을 1로 만들뿐, 형태는 유지한다[2,4,5,5]
        # 이부분이 사실상 가장 중요한 결과값으로 보인다.
        output = weights @ V # [2,4,5,5] @ [2,4,5,5] = [2,4,5,5]

        # [2,4,5,5].transpose(1,2) > [2,5,4,5].contiguous()[유지] > [2,5,4,5].view[2,-1,16] > [2,5,16]
        # contiguous? : 텐서가 메모리에 인접하게(contiguous) 저장되도록 보장하는 기능, 안전한 메모리에 의의를 두었다고하는데...
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # 형태는 다시 [2,5,16]으로 돌아왔다.

        # Linear[16,16]은 선형변환으로, 앞으 [2,5]는 영향을 주지않는다. 마지막 차원만 d_model로 출력(같음)
        print(output.shape)
        return self.W_o(output) # [2,5,16]

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

# Encoder Output
enc_output = torch.rand(batch_size, seq_len, d_model) # 2,5,16
# Decoder Hidden
dec_hidden = torch.rand(batch_size, seq_len, d_model) # 2,5,16

cross_attention = CrossAttention(d_model, num_heads) # 16, 4
# Q(dec_hidden), K(enc_output), V(enc_output)
result = cross_attention(dec_hidden, enc_output, enc_output)

print(f"입력(Decoder Hidden) : ${dec_hidden.shape}")
print(f"입력(Encoder Output) : ${enc_output.shape}")
print(f"출력 : ${result.shape}")