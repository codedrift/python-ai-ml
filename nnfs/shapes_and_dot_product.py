import numpy as np

print("NumPy version",np.__version__)

inputs = np.array([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])

print(inputs)

print(f'Inputs shape is {inputs.shape}')

weights = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
])

print(weights)

print(f'Weights shape is {weights.shape}')



biases = [2, 3, 0.5]

output = np.dot(inputs, weights.T) + biases

print("output:")
print(output)
print(f'Output has shape {output.shape}')

