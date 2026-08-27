import torch
print("PyTorch 版本:", torch.__version__)

a = torch.tensor(7)
print("张量 a 的形状:", a.shape)
print("张量 a 的数据类型:", a.dtype)

b = torch.tensor([1, 2, 3])
print("张量 b 的形状:", b.shape)
print("张量 b 的数据类型:", b.dtype)

c = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("张量 c 的形状:", c.shape)
print("张量 c 的数据类型:", c.dtype)

d = torch.tensor([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]])
print("张量 d 的形状:", d.shape)
print("张量 d 的数据类型:", d.dtype)

m1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
m2 = torch.tensor([[5, 6, 7], [7, 8, 9]])
m3 = m1 + m2
print("矩阵 m1+m2:", m3)
print("shape: ", m3.shape)

m4 = m1 * m2
print("矩阵 m1*m2:", m4)
print("shape: ", m4.shape)

x = torch.tensor([[1, 2, 3], [3, 4, 5]])
y = torch.tensor([[1, 2], [3, 4], [5, 6]])
z = torch.matmul(x, y)
print("矩阵 x*y:", z)
print("shape: ", z.shape)
