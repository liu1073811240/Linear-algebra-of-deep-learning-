import numpy as np
import torch

a = np.arange(1, 12, 2).reshape(2, 3)
# print(a)

b = a.T
# print(b)

c = np.arange(12).reshape(2, 2, 3)
# print(c)

a1 = torch.arange(12).reshape(3, 4)
print(a1)
# q = a1.T
q = a1.t()
print(q)


d = np.transpose(c, [2, 0, 1])  # 打乱内部顺序
# print(d.shape)
print(d)

e = np.reshape(c, [3, 2, 2])
print(e)


