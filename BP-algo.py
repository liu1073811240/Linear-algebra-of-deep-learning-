import random
import math


# 定义网络类
class Neural:

    # 从正太分布采样，获得w
    def __init__(self):
        self.w1 = random.normalvariate(0, 0.1)
        self.w2 = random.normalvariate(0, 0.1)
        self.b = 0

    # 前向输入计算
    def forward(self, x1, x2):
        nety = x1 * self.w1 + x2 * self.w2 + self.b
        return nety, self.outy(nety)  # 返回网络净输出和激活后输出

    # 后向输出计算
    def backward(self, x1, x2, outy, target):
        # 定义误差函数
        Etotal = (1 / 2) * (target - outy) ** 2
        print(Etotal)

        # 更新下一个W值
        self.w1 = self.w1 - 0.1 * self._outy(outy) * (target - outy) * x1  # 0.1步长，更新后的W1=当前W1减去误差对当前W1的求导结果
        self.w2 = self.w2 - 0.1 * self._outy(outy) * (target - outy) * x2  # 0.1步长，更新后的W2=当前W1减去误差对当前W2的求导结果
        self.b = self.b - 0.1 * self._outy(outy) * (target - outy)  # 0.1步长，更新后的b=当前W1减去误差对当前b的求导结果

    # 训练样本
    def train(self):
        # 实例化样本类
        sample = Sample()

        # 训练网络10000次
        for i in range(100000):
            x1, x2, target = sample.getSingle()  # 从随机取样里面获得输入和目标输出
            nety, outy = self.forward(x1, x2)  # 调用前向计算获得nety和outy
            self.backward(x1, x2, target, outy)  # 调用后向计算，算出损失（误差），更新W

    # 测试集网络
    def verify(self):
        print("-------------------------------------------------")
        # 给出测试集的输入样本，调用前向计算测试网络的outy和nety
        print(self.forward(0, 0))
        print(self.forward(1, 0))
        print(self.forward(0, 1))
        print(self.forward(1, 1))
        print(self.forward(1, 2))
        print(self.forward(2, 2))

    # 定义nety的sigmoid函数——outy，压缩数据在0和1之间
    def outy(self, nety):
        return 1 / (1 + math.exp(-nety))

    # 对outy函数求导，用以反向传播误差，更新W 的权重
    def _outy(self, nety):
        return self.outy(nety) * (1 - self.outy(nety))


# 定义样本类
class Sample:
    # 给出训练集的样本
    def __init__(self):
        # 解决或问题的输入和输出样本.
        self.s = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 1.]]

    # 对训练集样本随机取样
    def getSingle(self):
        return self.s[random.randint(0, 3)]


# 调用函数方法
if __name__ == '__main__':  # 主函数
    neural = Neural()  # 实例化对象
    neural.train()  # 调用训练集
    neural.verify()  # 调用测试集
