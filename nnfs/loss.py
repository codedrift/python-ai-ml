
import math

import numpy as np

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

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = np.array([
    [1,0,0], # dog, cat, cat
    [0,1,0],
    [0,1,0]
])  

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets    
    ]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)

# y_pre_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

print(np.e**(-np.inf))
print(average_loss)


softmax_outputs = np.array([
    [0.7, 0.2, 0.1],
    [0.5, 0.1, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = np.array([0,1,1])  

print("class_targets",class_targets)

predictions = np.argmax(softmax_outputs, axis=1)

if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions == class_targets)

print(predictions, accuracy)