import numpy as np
import torch

# 求行列式
a = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])  # 奇异矩阵
print(np.linalg.det(a))

b = torch.tensor([[1, 5, 7], [2, 5, 8], [3, 6, 9]], dtype=torch.float32)
print(torch.det(b))









