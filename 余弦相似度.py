import torch
import math

# 求向量的余弦相似度
a = torch.tensor([[0], [0.], [5.]])
b = torch.tensor([[5.], [0.], [0.]])
# b = torch.tensor([[0.], [0.], [5.]])

a1 = torch.sqrt(torch.sum(torch.pow(a, 2)))
b1 = torch.sqrt(torch.sum(torch.pow(b, 2)))

cos_theta = torch.matmul(a.t(), b) / (a1*b1)
print(cos_theta)

theta_r = torch.acos(cos_theta)  # 弧度
# print(theta_r)

theta_a = math.degrees(theta_r)  # 弧度转角度
print(theta_a)


