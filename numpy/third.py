import numpy as np

layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
]

print(layer_outputs)

exp_values = np.exp(layer_outputs - np.max(layer_outputs), dtype=np.float32)

print(exp_values)

layer_sum = np.sum(exp_values, axis=1, keepdims=True)

norm_values = exp_values / layer_sum

print(norm_values)