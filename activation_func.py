import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

x = np.arange(-10, 10, 0.1)
# print(x)

# 1/(1+e^-x)
y1 = 1/(1+np.exp(-x))

# (e^x-e^-x) / (e^x+e^-x)
y2 = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()


