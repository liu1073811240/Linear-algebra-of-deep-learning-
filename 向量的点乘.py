import torch
import numpy as np

#torch.dot只能做向量的点乘，无法做矩阵乘法
a = torch.arange(3)
b = torch.arange(3)
a1 = torch.arange(6).reshape(3, 2)
b1 = torch.arange(3).reshape(1, 3)

print(a*b)
print(a@b, b@a1)
print(torch.dot(a, b))
# print(torch.dot(b1, a1))  # 不能做矩阵乘法
print(torch.matmul(a, b))
print(torch.matmul(b1, a1))

c = np.arange(6).reshape(3, 2)
d = np.arange(3).reshape(1, 3)
c1 = np.arange(3)
d1 = np.arange(3)
print(np.dot(d, c))
print(np.dot(c1, d1))
print(np.dot(d1, c1))
print(d@c)
print(np.matmul(d, c))

