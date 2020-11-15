import numpy as np
import matplotlib.pyplot as plt
import random

'''创建线性分类器'''

x = np.array([0, 20]).reshape(2, 1)
y = np.array([20, 0]).reshape(2, 1)


class Net:
    def __init__(self, a, b, c, d):
        self.w1 = np.random.normal(0, 0.01, (a, b)) # [2, 10]
        self.b1 = np.zeros(b)  # [10]
        self.w2 = np.random.normal(0, 0.01, (b, c))  # [10, 100]
        self.b2 = np.zeros(c)  # [100]
        self.w3 = np.random.normal(0, 0.01, (c, d))  # [100, 1]
        self.b3 = np.zeros(d)  # [1]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):  # [2, 1]
        # 创建网络前向
        # [2,1]@[1,10]+[10]=[2,10]
        self.y1 = np.dot(x, self.w1) + self.b1  # [2, 10]
        # [2,10]@[10,100]+[100] = [2,100]
        self.y2 = np.dot(self.y1, self.w2) + self.b2 # [2, 100]
        # [2, 100]@[100,1]+[1] = [2,1]
        self.y3 = np.dot(self.y2, self.w3) + self.b3  # [2, 1]

        return self.y3

    def optimizer(self, output, label, wlr, blr):
        # 计算梯度
        # output = f(x) = ((w1x+b1)*w2+b2)w3+b3
        # y1 = w1x+b1, y2 = y1*w2+b2, y3 = y2*w3+b3
        # y3 = output
        # loss = 1/2(y3-y)**2
        # dloss/dw3 = (dloss/dy3)*(dy3/dw3)
        # dloss/db3 = (dloss/dy3)*(dy3/db3)
        dw3 = np.mean(-(output - label) * self.y2)
        db3 = np.mean(-(output - label) * 1)

        # dloss/dw2 = (dloss/dy3)*(dy3/dy2)*(dy2/dw2)
        # dloss/dw2 = (dloss/dy3)*(dy3/dy2)*(dy2/db2)
        dw2 = np.mean(-(output - label) * np.mean(self.w3) * self.y1)
        db2 = np.mean(-(output - label) * 1)

        # dloss/dw1 = (dloss/dy3)*(dy3/dy2)*(dy2/dy1)*(dy1/w1)
        # dloss/dw1 = (dloss/dy3)*(dy3/dy2)*(dy2/dy1)*(dy1/b1)
        dw1 = np.mean(-(output - label) * np.mean(self.w3) * np.mean(self.w2) * x)
        db1 = np.mean(-(output - label) * 1)

        # 反向更新参数
        self.w3 = self.w3 + wlr * dw3
        self.b3 = self.b3 + blr * db3
        self.w2 = self.w2 + wlr * dw2
        self.b2 = self.b2 + blr * db2
        self.w1 = self.w1 + wlr * dw1
        self.b1 = self.b1 + blr * db1


class MSE:
    def lossfunction(self, output, label):
        return np.mean(1 / 2 * (output - label) ** 2)

    def __call__(self, output, label):
        return self.lossfunction(output, label)


if __name__ == '__main__':
    net = Net(1, 10, 100, 1)
    loss_func = MSE()

    plt.ion()  # 动态画图
    for i in range(10000):
        output = net(x)
        loss = loss_func(output, y)
        net.optimizer(output, y, 0.01, 0.01)

        if i % 5 == 0:
            plt.clf()
            print(loss)
            a = np.random.uniform(0, 10, 100)
            b = np.random.uniform(0, 10, 100)
            c = np.random.uniform(10, 20, 100)
            d = np.random.uniform(10, 20, 100)

            plt.scatter(a, b)
            plt.scatter(c, d)
            plt.plot(x, y)
            plt.plot(x, output)
            plt.text(0, 0, "loss:{:.3f}".format(loss), fontdict={"size":20, "color":"red"})

            plt.pause(0.01)

    plt.ioff()
    plt.show()





