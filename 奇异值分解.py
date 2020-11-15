import torch
import numpy as np

# 奇异值分解：把一个矩阵拆分成三个矩阵

a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
b = np.array([[1., 2.], [3., 4.], [5., 6.]])

# 3*2: 3*2, 2*2(奇异值：对角阵)， 2*2
print(torch.svd(a))

# 3*2: 3*3, 3*2(奇异值：对角阵，不足填充0)， 2*2
print(np.linalg.svd(b))



