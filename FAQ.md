
## Was macht optimizer=adam?

Adam is different to classical stochastic gradient descent.

Stochastic gradient descent maintains a single learning rate (termed alpha) for all weight updates and the learning rate does not change during training.

A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.

The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

The authors describe Adam as combining the advantages of two other extensions of stochastic gradient descent. Specifically:

**Adaptive Gradient Algorithm (AdaGrad)** that maintains a per-parameter learning rate that improves performance on problems with sparse gradients (e.g. natural language and computer vision problems).

**Root Mean Square Propagation (RMSProp)** that also maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight (e.g. how quickly it is changing). This means the algorithm does well on online and non-stationary problems (e.g. noisy).

[Source](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20a%20replacement%20optimization,sparse%20gradients%20on%20noisy%20problems.)

## Was macht activation=relu?
Für alle inputs unter 0 gibt es 0 raus und über 0 gibt es den input raus

## Was macht loss=binary_crossentropy?
Binary crossentropy is a loss function that is used in binary classification tasks. These are tasks that answer a question with only two choices (yes or no, A or B, 0 or 1, left or right).

Sigmoid is the only activation function compatible with the binary crossentropy loss function. You must use it on the last block before the target block.

[Source](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy)
## Was macht activation="sigmoid"?

Applies the sigmoid activation function. For small values (<-5), sigmoid returns a value close to zero, and for large values (>5) the result of the function gets close to 1.

[Source](https://keras.io/api/layers/activations/)

## Wie weiss ich wieviele Neuronen ich pro Layer nehmen sollte?

[Beginners Ask “How Many Hidden Layers/Neurons to Use in Artificial Neural Networks?”](https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e)

## Wie weiss ich wieviele Layer ich nehmen muss?

[How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)

## Was gibt es noch für activation functions und wofür brauch ich die?

[An overview of activation functions used in neural networks](https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html#:~:text=An%20activation%20function%20is%20used,in%20an%20artificial%20neural%20network.&text=Non%2Dlinearity%20means%20that%20the,network%20becomes%20a%20universal%20approximator.)


## Was gibt es noch für optimizer functions und wofür?

[Various Optimization Algorithms For Training Neural Network](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6#:~:text=Optimizers%20are%20algorithms%20or%20methods,order%20to%20reduce%20the%20losses.&text=How%20you%20should%20change%20your,by%20the%20optimizers%20you%20use.)
## Was gibt es noch loss functions und wofür?


[Common Loss functions in machine learning](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23#:~:text=It's%20a%20method%20of%20evaluating,reduce%20the%20error%20in%20prediction.)
## Was ist Hyperparameter-Tuning?
 A hyperparameter is a parameter whose value is used to control the learning process. 

 [Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
