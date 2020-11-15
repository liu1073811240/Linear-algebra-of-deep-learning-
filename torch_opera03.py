import torch
import numpy as np


a = torch.arange(6).reshape(2, 3)
b = torch.arange(6).reshape(3, 2)
# a = np.arange(6).reshape(2, 3)
# b = np.arange(6).reshape(3, 2)
c = torch.arange(3)
d = torch.arange(6).reshape(2, 3)
a1 = torch.arange(6)
b1 = torch.arange(1, 7)

print(a1)
print(b1)
# print(a.dot(b))  # 向量和向量的点乘，(torch只能在一维中进行，二维中就不可以，而在numpy中却可以)

print(a@b)  # 矩阵相乘
print(a.matmul(b))
print(torch.matmul(a, b))


