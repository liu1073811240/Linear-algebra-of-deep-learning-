import numpy as np

x = np.array([[3], [1], [6]])
y = 4*x

print(x)
print(y)
# 最小二乘法
print(np.linalg.inv((x.T@x))@x.T@y)
print(np.matrix(x.T@x).I@x.T@y)




