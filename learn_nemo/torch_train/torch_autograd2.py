import torch

x = torch.tensor([2.0])
x.requires_grad_(True)

w = torch.tensor([3.0])
w.requires_grad_(True)

y = x * w + w **2
print("张量 y:", y)
y.backward()
print("张量 x 的梯度:", x.grad)
print("张量 w 的梯度:", w.grad)
x.grad.zero_()
w.grad.zero_()

b = torch.tensor([10.0])
z = x * w + b
print("张量 z:", z)
z.backward()
print("张量 x 的梯度:", x.grad)
print("张量 w 的梯度:", w.grad)
