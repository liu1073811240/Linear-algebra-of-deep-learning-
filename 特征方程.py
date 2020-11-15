import torch

# vAx = vλx
# 等式左边:变换矩阵特征值(对角线值)要等于λ
a = torch.tensor([1., 2.])
b = torch.tensor([[2., 0.], [0., 2.]])
c = torch.tensor([3., 4.])
print(a*b*c)

# 等式右边：
d = torch.tensor([1., 2.])
e = torch.tensor([2.])
f = torch.tensor([3., 4.])
print(d*e*f)
