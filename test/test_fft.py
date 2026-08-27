import torch

# 10x10 的黑白相间竖线条纹：偶数列为 1，奇数列为 -1
x = torch.ones(10, 10, dtype=torch.float32)
x[:, 1::2] = -1

print("输入形状:", x.shape)
print("输入矩阵:\n", x)

# 2D 实数 FFT
X_freq = torch.fft.rfft2(x)
print("输出形状:", X_freq.shape)
print("输出类型:", X_freq.dtype)
print("频谱矩阵（实部+虚部）:\n", X_freq)

# 只看幅值
mag = torch.abs(X_freq)
print("幅值矩阵:\n", mag)

