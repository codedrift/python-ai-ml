
import math

import numpy as np

from dense import (Activation_Softmax,
                   Activation_Softmax_Loss_CategoricalCrossentropy, Loss_CategoricalCrossEntropy)

from timeit import timeit

# # probability distribution => categorical cross entropy
# softmax_output = [0.7, 0.1, 0.2]

# # one hot!
# target_output = [1, 0, 0]

# loss = -(
#     math.log(softmax_output[0]) * target_output[0] +
#     math.log(softmax_output[1]) * target_output[1] +
#     math.log(softmax_output[2]) * target_output[2]
# )

# one_hot_loss = -math.log(softmax_output[0])


# print(loss, one_hot_loss)

# print(math.log(1))
# print(math.log(0.5))
# print(math.e ** math.log(5.2))
# print(10 ** 2)

# softmax_outputs = np.array([
#     [0.7, 0.1, 0.2],
#     [0.1, 0.5, 0.4],
#     [0.02, 0.9, 0.08]
# ])

# class_targets = np.array([
#     [1,0,0], # dog, cat, cat
#     [0,1,0],
#     [0,1,0]
# ])  

# if len(class_targets.shape) == 1:
#     correct_confidences = softmax_outputs[
#         range(len(softmax_outputs)),
#         class_targets    
#     ]
# elif len(class_targets.shape) == 2:
#     correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

# neg_log = -np.log(correct_confidences)

# average_loss = np.mean(neg_log)

# # y_pre_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

# print(np.e**(-np.inf))
# print(average_loss)


softmax_outputs = np.array([
    [0.7, 0.2, 0.1],
    [0.5, 0.1, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = np.array([0,1,1])

def f1():
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.backward(softmax_outputs,class_targets)
    dvalues1 = softmax_loss.dinputs

def f2():
    activation = Activation_Softmax()
    activation.output = softmax_outputs
    loss = Loss_CategoricalCrossEntropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs

# print("Gradients: combined loss and activation:")
# print(dvalues1)
# print("Gradients: separate loss and activation:")
# print(dvalues2)

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)

print(t2)
print(t2/t1)

# print("class_targets",class_targets)

# predictions = np.argmax(softmax_outputs, axis=1)

# if len(class_targets.shape) == 2:
#     class_targets = np.argmax(class_targets, axis=1)

# accuracy = np.mean(predictions == class_targets)

# print(predictions, accuracy)
