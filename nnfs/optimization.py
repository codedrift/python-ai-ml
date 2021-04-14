import matplotlib
import matplotlib.pyplot as plt
import nnfs

from nnfs.datasets import vertical_data
# matplotlib.use('TkAgg')
matplotlib.use('tkagg')

nnfs.init()

X, y = vertical_data(samples=100, classes=3)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
