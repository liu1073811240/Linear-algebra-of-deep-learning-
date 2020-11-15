import numpy as np
import torch

# 轴交换
c = np.arange(12).reshape([2, 2, 3])
# print(c)
c1 = torch.arange(12).reshape([2, 2, 3])
print(c1)

d1 = torch.transpose(c1, 2, 0)
d2 = torch.transpose(d1, 1, 2)
print(d2)
d3 = c1.permute([2, 0, 1])
print(d3)


