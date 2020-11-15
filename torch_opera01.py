import torch

# 矩阵的加减法
a = torch.arange(6).reshape(2, 3)
b = torch.arange(2).reshape(2, 1)
c = torch.arange(3).reshape(1, 3)
d = torch.arange(6).reshape(2, 3)
e = torch.tensor(2)


print(a)
print(b)
print(a+b)
print(c)
print(a+c)
print(a+d)



