import numpy as np

# 求内积，外积
a = np.array([1, 2])
b = np.array([3, 4])

c = np.sum(a*b)
print(c)
print(np.dot(a, b))
print(a*b)

