import numpy as np
import torch

# 行列式
a = np.array([[1, 2], [3, 4]])
print(np.linalg.det(a))

b = torch.tensor([[1., 2.], [3., 4.]])
print(b.det(), torch.det(b))

c = torch.tensor([[1., 4., 7], [2., 5., 8], [3., 6., 9]])
print(c.det())

d = torch.tensor([[1., 5., 7], [2., 5., 8], [3., 6., 9]])
print(d.det())
