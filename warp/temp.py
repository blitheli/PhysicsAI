import warp as wp
import numpy as np

a1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

w1 = wp.array(a1, dtype=wp.vec3)

a12 = w1.numpy()

cc = [a12]

print(cc[0].shape)
print(cc[0])
