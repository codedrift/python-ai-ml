# import matplotlib.pyplot as plt
import numpy as np

# import nnfs
# optional: use spiral data nnfs package
from nnfs.datasets import spiral_data

# this sets some variables like random seeds to be consistent on different machines
# https://github.com/Sentdex/nnfs/blob/master/nnfs/core.py
# nnfs.init()

# set numpy random seed manually
# np.random.seed(0)

# https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py


def spiral_data(samples, classes):
    print(f"Generating spriral data samples: {samples}, classes: {classes}")
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, samples)
            + np.random.randn(samples) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


# https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/vertical.py


def vertical_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        X[ix] = np.c_[
            np.random.randn(samples) * 0.1 + (class_number) / 3,
            np.random.randn(samples) * 0.1 + 0.5,
        ]
        y[ix] = class_number
    return X, y


class Layer_Dense:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_regularizer_l1=0,
        weight_regularizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # backward pass
    def backward(self, dvalues):

        # create uninitialized array for resulting gradients
        self.dinputs = np.empty_like(dvalues)
        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # flatten array
            single_output = single_output.reshape(-1, 1)
            # calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# we calc the derivate of categorical crossentropy  with respect to the activation
# function to save on computation
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    # backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in abatch
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

    def regularization_loss(self, layer: Layer_Dense):
        regularization_loss = 0

        # l1 reg weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(
                np.abs(layer.weights)
            )

        # l2 reg weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(
                layer.weights * layer.weights
            )

        # l1 reg bias
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(
                np.abs(layer.biases)
            )

        # l2 reg biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(
                layer.biases * layer.biases
            )

        return regularization_loss


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer: Layer_Dense):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSProp:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        )
        layer.bias_cache = (
            self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        )

        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    def __init__(
        self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )

        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )

        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        )

        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        )

        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )

        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


# generate some data in a spiral with three classes
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# optimizer = Optimizer_SGD(learning_rate=1, decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    # first layer
    dense1.forward(X)

    # pass data to first activation function
    activation1.forward(dense1.output)

    # second dense layer
    dense2.forward(activation1.output)

    # pass output to combined loss and activation
    # first data is passed to the softmax activation
    # second the loss is calculated with the optimized derivate
    loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.reg


    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            f"epoch: {epoch} acc: {accuracy:.3f} loss: {loss:.3f} lr: {optimizer.current_learning_rate}"
        )

    # BACKWARD PASS
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)

if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f"Validation, acc: {accuracy:.3f}, loss: {loss:.3f}")
