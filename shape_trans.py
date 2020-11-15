import numpy as np
import torch

# 形状类型查看
a = np.array([[1, 2], [3, 4]])
print(a.shape, np.shape(a), a.dtype)

a = a.astype(np.float32)
print(a.dtype)

b = torch.tensor([[1, 2], [3, 4]])
print(b.shape, b.size(), b.dtype)
print(b.dtype)



