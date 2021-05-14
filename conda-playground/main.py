import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import type_check


def f(x):
    return 2*x**2


x = np.arange(0,5, 0.001)
y = f(x)

plt.plot(x, y)

# plt.show()



colors = ["k","g", "r", "b", "c"]



def approximate_tanget_line(x, approximate_derivate):
    return approximate_derivate * x + b

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1,x2), (y1,y2))

    approximate_derivate = (y2 - y1) / (x2 - x1)
    b = y2 - approximate_derivate*x2

    to_plot = [x1-0.9,x1, x1+0.9]
    plt.scatter(x1,y1, c=colors[i])

    plt.plot([plot for plot in to_plot], [approximate_tanget_line(point, approximate_derivate) for point in to_plot], c=colors[i])
    print("Approximate derivate for f(x)", f"where x = {x1} is {approximate_derivate}")


plt.show()
