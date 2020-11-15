import numpy as np
import torch

# 对角阵
a = np.diag([1, 2, 3, 4])
print(a)

b = torch.diag(torch.tensor([1, 2, 3, 4]))
print(b)
print(b.T)

# 单位矩阵
c = np.eye(3, 4)
print(c)

d = torch.eye(3, 4)
print(d)

# 下三角矩阵
e = np.tri(3, 3)
print(e)
f = torch.tril(torch.ones(3, 3))
print(f)

# 0, 1矩阵
# g = np.ones((3, 3))
g = np.zeros((3, 3))
print(g)

# h = torch.ones(3, 3)
h = torch.zeros(3, 3)
print(h)

