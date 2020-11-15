import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[2., 0.], [0., 2.]])

# 求特征值
print(torch.eig(a))

print()
# 特征向量（轴）
print(torch.eig(a, eigenvectors=True))
print(torch.eig(b, eigenvectors=True))





