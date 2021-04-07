import matplotlib.pyplot as plt

import numpy as np

np.random.seed(0)

#https://cs231n.github.io/neural-networks-case-study/
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# X = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]
# ]

# layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)






X,y = spiral_data(points=100,classes=3)

print("spiral data shape",X.shape, y.shape)

# plt.style.use('dark_background')
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("activation2 shape", activation2.output.shape)
print(activation2.output[:5])