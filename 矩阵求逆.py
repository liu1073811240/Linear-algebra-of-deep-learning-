import  torch
import numpy as np

a = torch.tensor([[1., 2.], [3., 4.],])
# b = a.t()
# print(a@b)

# 逆矩阵
print(torch.inverse(a))

b = np.array([[1., 2.], [3., 4.]])
print(np.linalg.inv(b))

c = np.matrix(np.array([[1., 2.], [3., 4.]]))
print(c.I)
