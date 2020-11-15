import torch

# 数乘（叉乘）
a = torch.arange(6).reshape(2, 3)
b = torch.arange(2).reshape(2, 1)
c = torch.arange(1, 3)
d = torch.arange(6).reshape(2, 3)
e = torch.tensor(2)


print(a)
print(b)
print(c)
print(d)
print(e)

print("---------------")
print(b*e)
print(b*c)
print(a*b)
print(a*d)

