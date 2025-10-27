import torch
import torch.nn as nn
import torch.optim as optim

linear = nn.Linear(5,2) # (in_features, out_features)

x = torch.randn(3,5) # (batch_size, in_features)
y = linear(x)

criterion = nn.MSELoss()

# torch.optim : 최적화 알고리즘 집합 모듈
# torch.optim.SGD : Stochastic Gradient Descent(확률적 경사 하강법)
# linear.parameters() : <bound method Module.parameters of Linear(in_features=5, out_features=2, bias=True)>
# lr(Learning Rate) : 최적화 과정에서 지정된 수치만큼 매개변수가 '이동'되는 비율
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
x = torch.randn(10,5)
target = torch.randn(10,2)

output = linear(x)
loss = criterion(output, target)
loss.backward() # 역전파 실행
optimizer.step()

model = nn.Linear(1, 1)  # 입력1, 출력1

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# [1,4]
x_train = torch.tensor([
    [1.0], [2.0], [3.0], [4.0]
])

y_train = 2 * x_train + 1  # y = 2x + 1

for epoch in range(1000):
    # Forward
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if (epoch + 1) % 100 == 0:
        w, b = model.weight.item(), model.bias.item()
#        print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.6f} | w={w:.4f}, b={b:.4f}")

# 결과 확인
#print("\n최종 학습된 파라미터:")
#print("가중치(w):", model.weight.item())
#print("편향(b):", model.bias.item())

# Linear initialize

layer = nn.Linear(4, 2)
#print(f"Layer\n{layer}")
#print(f"Weight : {layer.weight}\nbias : {layer.bias}")

w = torch.empty(4,2)

k_u = nn.init.kaiming_uniform_(w)

#print(f"kaiming_uniform_ [4,2]\n{k_u}")

nn.init.constant_(layer.weight, 0.5)
#print(layer.weight)
nn.init.zeros_(layer.bias)

nn.init.xavier_uniform_(layer.weight)
#print(layer.weight)
nn.init.zeros_(layer.bias)

nn.init.kaiming_uniform_(layer.weight)
#print(layer.weight)
nn.init.zeros_(layer.bias)

x = torch.randn(2,3,requires_grad=True)
W = torch.randn(4,3,requires_grad=True)
b = torch.randn(4, requires_grad=True)

y = x @ W.T + b # [2,3]@[3,4] = [2,4]

loss = y.sum()
loss.backward()

print(f"W.T\n{W.T}")

print(f"X\n{x}")
print(f"X.grad\n{x.grad}")
print(f"Y\n{y}")
print(f"Y.grad\n{y.grad}") # grad=None
print(f"W\n{W}")
# requires_grad=False로 하면 None 이나온다.
print(f"W.grad\n{W.grad}")
print(f"b\n{b}")
# requires_grad=False로 하면 None 이나온다.
print(f"b.grad\n{b.grad}") # 얘는 항상 [1,4]의 2 임