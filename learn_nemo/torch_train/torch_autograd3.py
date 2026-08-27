import torch

v = torch.tensor([1.0, 2.0, 3.0])
v.requires_grad_(True)

s = v.sum()
print("张量 s:", s)
s.backward()
print("张量 v 的梯度:", v.grad)