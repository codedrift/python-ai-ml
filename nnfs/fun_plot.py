import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mpl_lines


def f(x):
    return 2*x**2

SIZE = 10

x = np.array(range(SIZE))
y = f(x)

print("x", x)
print("y", y)

# plt.plot(x, y)


m = []

for i in range(SIZE - 1):
       m.append((y[i + 1] - y[i]) /(x[i + 1] - x[i]))

print("m",m)

fig = plt.figure()
ax = plt.axes()

# base graph
ax.plot(x, f(x))

for i in range(SIZE - 3):
    new_y = m[i] * x - m[i]
    better_y = new_y - (new_y[i] - y[i])
    ax.plot(x, better_y)
    print(i, x[i:i +2], better_y[i:i + 2], y[i:i + 2])
    ax.plot(x[i:i +2], better_y[i:i + 2], 'bo')

plt.show()
