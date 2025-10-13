import torch
import torch.nn as nn
import matplotlib.pyplot as plt

d_model = 12
d_ff = 32

ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model),
)

x = torch.rand(12,d_model)

print(f"Input : ${x.size()} \n${x}")

y_1 = ffn[0](x)
y_2 = ffn[1](y_1)
y_3 = ffn[2](y_2)


print(f"Output 1 : ${y_1.size()} \n${y_1}") # fc1
print("")
print(f"Output 2 : ${y_2.size()} \n${y_2}") # fc1+ReLU
print("")
print(f"Output 3 : ${y_3.size()} \n${y_3}") # fc1+ReLU+fc2

# 음수는 Fully Connected 에서 발생한다.