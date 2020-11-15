import random
import matplotlib.pyplot as plt
import torch
from torch import nn

# 利用pytorch实现线性回归
x = torch.linspace(-1, 1, 100).reshape([100, 1])
w = torch.rand(x.size())
b = torch.rand(x.size())

y = w*x + b

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
    loss_fun = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=0.005)

    plt.ion()
    for i in range(100):

        z = net(x)
        loss = loss_fun(z, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss)
        plt.clf()
        plt.plot(x.detach(), z.detach())
        # plt.scatter(x, y)
        # plt.scatter(x.detach(), z.detach())
        plt.scatter(x.detach(), y.detach())


        plt.pause(0.01)

    plt.ioff()
    plt.show()










