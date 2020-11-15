import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# 利用pytorch实现线性分类器

x = np.array([0, 20]).reshape(2, 1)
y = np.array([20, 0]).reshape(2, 1)

class Net(nn.Module):
    def __init__(self, a, b, c, d):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(a, b),
            nn.Linear(b, c),
            nn.Linear(c, d)

        )

    def forward(self, x):
        self.y = self.conv(x)
        return self.y

if __name__ == '__main__':

    net = Net(1, 10, 100, 1)
    loss_func = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), 0.0001)
    # for params in net.parameters():
    #     print(params)
    # exit()

    plt.ion()  # 动态画图
    for i in range(10000):
        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(y, dtype=torch.float32)

        output = net(x)
        loss = loss_func(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

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
            plt.plot(x.detach(), output.detach())
            plt.text(0, 0, "loss:{:.3f}".format(loss.item()), fontdict={"size": 20, "color": "red"})

            plt.pause(0.01)

    plt.ioff()
    plt.show()

















