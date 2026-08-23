import numpy as np
import warp as wp
import os

if wp.get_cuda_device_count() > 0:
    print("------------------------------------------------")    
    print("CUDA device count:", wp.get_cuda_device_count())
    print("kernel cache dir:", wp.config.kernel_cache_dir)

# 更改 Warp 内核缓存目录，避免与其他示例冲突
wp.config.kernel_cache_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "warp_cache"
)

@wp.func
def f(x: wp.float64) -> wp.float64:
    return wp.sin(x*x)*wp.log(x)+ x*x*x/wp.sqrt(wp.float64(1.0)-x*x)

@wp.kernel
def celsius_to_fahrenheit(celsius: wp.array[float], fah: wp.array[float]):

    i = wp.tid()  # 获取当前线程的索引

    fah[i] = celsius[i] * 9.0 / 5.0 + 32.0

    c3 = f(wp.float64(0.2) * wp.float64(i + 1))  # 给 f 传入 (0, 1) 区间内的安全输入


celsius = wp.array([0.0, 20.0, 37.0, 100.0], dtype=float)
fah = wp.zeros(4, dtype=float)

wp.launch(celsius_to_fahrenheit, dim=4, inputs=[celsius, fah])

print(f"Celsius: {celsius}")
print(f"Fahrenheit: {fah}")
print("-------------------------------")

@wp.kernel
def compute(x: wp.array[wp.float64], y: wp.array[wp.float64]):

    i = wp.tid()  # 获取当前线程的索引

    y[i] = f(x[i])  # 调用自定义函数 f，传入当前线程索引 i 对应的 x 数组元素

x = wp.array([0.5, 0.75], dtype=wp.float64, requires_grad=True)
y = wp.zeros(2, dtype=wp.float64, requires_grad=True)
y_seed = wp.array([1.0, 0.0], dtype=wp.float64)

with wp.Tape() as tape:
    wp.launch(compute, dim=2, inputs=[x, y])

tape.backward(grads={y: y_seed})
print(f"y: {y.numpy()}")
print(f"x.grad: {x.grad.numpy()}")
print("------------End----------------")

