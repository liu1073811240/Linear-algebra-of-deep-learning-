import torch

a = torch.randn([2, 3, 2])
print(a)

b = a.reshape(3, 4)
print(b)

c = a.reshape(1, 12)
print(c)

d = a.reshape(12, 1)
print(d)

e = a.reshape(12)
print(e)
# e = a.unsqueeze(0)
t = torch.unsqueeze(e, 1)
print(t)

g = torch.squeeze(t, 1)
print(g)
