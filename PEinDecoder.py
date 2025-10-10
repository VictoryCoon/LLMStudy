import numpy as np
import matplotlib.pyplot as plt

def posisional_encoding(pos, d_model):
    PE = np.zeros((pos,d_model))
    for p in range(pos):
        for i in range(0,d_model,2):
            PE[p, i]   = np.sin(p / (10000 ** (2 * i / d_model)))
            PE[p, i+1] = np.cos(p / (10000 ** (2 * i / d_model)))
    return PE

# -------------------------
# 1. Positional Encoding 생성
# -------------------------
pos = 512
d_model = 16
PE = posisional_encoding(pos, d_model)

plt.figure(figsize=(24,12))
plt.pcolormesh(PE, cmap='RdBu')
plt.xlabel('Embedding Dimensions')
plt.ylabel('Position (word index)')
plt.title("Positional Encoding Pattern")
plt.colorbar()
plt.show()
"""
위의 차트에서 Embedding Dimensions이 1, 2수준일때는 Position index가 요동치는 수준으로 격차를 보인다.
차원 3 수준부터는 부드럽게 펴지는 형태로 전환되며,
4수준이 넘어간 후로 Position Index는 0아니면 100이되는 것을 확인할 수 있는데
이 단계부터 학습은 Boolean 수준이 된다는 것일까?
"""

# -------------------------
# 2. 단어 Embedding + PE
# -------------------------
# 임의의 Embedding (5개 단어, 16차원)
word_embeddings = np.random.rand(5,d_model) # 5 X 16
# PE
PE_subset = PE[:5,:]

combined = word_embeddings + PE_subset

print(f"Word Embeddings Shape : ${word_embeddings.shape}")
print(f"PE Shape : ${PE_subset.shape}") # 5 X ? > 16, 아래 내용을 연산하려면 같은 행렬 형태
print(f"Combined Shape : ${combined.shape}") # 5 X 16

# -------------------------
# 3. 시각화: 위치별 차이
# -------------------------
plt.figure(figsize=(24,12))
for i in range(5):
    plt.plot(combined[i],label=f"Token {i}")

plt.legend()
plt.title("Word Embedding + Positional Encoding")
plt.xlabel("Embedding Dimention")
plt.ylabel("Value")
plt.show()
"""
Embeding 어휘와 PE의 행렬합은 일전에 PE에 입력된 차원의 급격한 변화와 달리
0-2차원까지는 난잡하지만, 2차원 이후로는 홀수차원(cos)은 Value가중이 증가하고,
짝수차원(sin)은 Value가중이 감소하는 형태를 반복적으로 보이고 있다.
동시에 모든 Token들은 조금씩 거리가 수렴해간다.
이 현상 자체에 대해서는 잘 모르겠으나, '학습'이라는 관점으로 볼때 올바른 형태로 가는 듯 하다.
"""