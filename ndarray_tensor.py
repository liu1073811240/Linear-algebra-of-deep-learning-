import torch
import numpy as np


a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(a, type(a), a.dtype)

b = np.array([[1, 2], [3, 4]])
print(b, type(b), b.dtype)

c = a.numpy()
print(c)

d = torch.from_numpy(b)
print(d, type(d), d.dtype)




