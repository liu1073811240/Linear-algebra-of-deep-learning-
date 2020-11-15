import random
import matplotlib.pyplot as plt

# 线性回归
_x = [i/100 for i in range(100)]
_y = [3*j+4+random.random() for j in _x]

w = random.random()
b = random.random()
plt.ion()

for i in range(100):
    for x, y in zip(_x, _y):
        z = w*x + b
        o = z - y
        loss = 1/2*o**2
        dw = -o*x
        db = -o*1
        w = w + 0.1*dw
        b = b + 0.1*db
        print(w, b, loss)

        plt.clf()
        # plt.plot(_x, _y)
        plt.scatter(_x, _y)

        v = [w*e+b for e in _x]
        # plt.plot(_x, v)
        plt.scatter(_x, v)

        plt.pause(0.01)

plt.ioff()
plt.show()










