import torch
print("PyTorch 版本:", torch.__version__)

v = torch.tensor([1,2,3])
print(v[1])
v1 = v + 10
print("张量 v1:", v1)
print("张量 v1 的形状:", v1.shape)
print("张量 v1 的数据类型:", v1.dtype)

m = torch.tensor([[1, 2, 3], [4, 5, 6]])
mv = m + torch.tensor([10, 20, 30])
print("矩阵 m + 张量 [10, 20, 30]:", mv)
print("矩阵 mv 的形状:", mv.shape)

col = torch.tensor([[1], [2]])
row = torch.tensor([[10, 20, 30]])
broadcasted_sum = col + row
print("广播后的和:", broadcasted_sum)
print("广播后的和的形状:", broadcasted_sum.shape)

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[10, 11], [20, 21], [30, 31]])
broadcasted_sum2 = a + b
print("广播后的和2:", broadcasted_sum2)
print("广播后的和2的形状:", broadcasted_sum2.shape)