import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# FFN 설정 (입력 코드와 동일)
d_model = 4
d_ff = 16

ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model),
)

# 입력 텐서 생성
x = torch.rand(3, d_model)

# 각 레이어의 출력을 단계별로 저장
with torch.no_grad():  # 계산 그래프를 추적하지 않아 메모리 절약

    # 1. Linear (첫 번째 레이어)
    fc1_out = ffn[0](x)

    # 2. ReLU
    relu_out = ffn[1](fc1_out)

    # 3. Linear (두 번째 레이어)
    final_out = ffn[2](relu_out)

# 각 단계의 텐서 값을 flatten하여 numpy 배열로 변환
data = {
    '1. Input (x)': x.numpy().flatten(),
    '2. After Linear Layer 1': fc1_out.numpy().flatten(),
    '3. After ReLU Activation': relu_out.numpy().flatten(),
    '4. After Linear Layer 2 (Output)': final_out.numpy().flatten()
}

# 히스토그램 시각화
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Feed Forward Network (FFN) Data Flow', fontsize=16)

colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
for i, (title, values) in enumerate(data.items()):
    axs[i].hist(values, bins=30, color=colors[i], edgecolor='black')
    axs[i].set_title(title)
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()