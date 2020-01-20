import random

import numpy as np

# The default activation function that the neural network uses, as lambdas
sigmoid = lambda x: x / (1 + abs(x))
d_sigmoid = lambda x: 1 / (1 + pow(abs(x), 2))


class Main:
    # Initialize the neural network with all of the required variables
    def __init__(self, layers, learning_rate=.1, activation_function=sigmoid, d_activation_function=d_sigmoid):
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function
        self.net = [None] * (len(layers) - 1)
        self.out = [None] * (len(layers) - 1)

        # Make both the weights and the biases random
        for i in range(1, len(layers)):
            self.weights.append(np.random.rand(layers[i - 1], layers[i]))
            self.biases.append(np.random.rand(1, layers[i]))

    # Feeds forward the inputs through the neural network
    def feed_forward(self, inputs):
        result = [np.array(inputs)]
        for i in range(len(self.weights)):
            # Multiply the previous output with the current weights and add the bias
            x = result[len(result) - 1].dot(self.weights[i])
            x = np.add(x, self.biases[i])

            # Save both the output and the output without the activation function
            self.net[i] = x
            self.out[i] = self.activation_function(x)
            result.append(self.out[i])

    # The backward propagation
    def backward_propagation(self, targets, inputs):
        deltas = [None] * len(self.weights)
        # Calculate the delta of the output layer
        deltas[-1] = (targets - self.out[-1]) * self.d_activation_function(self.net[-1])

        # Calculate the deltas of the hidden layers using the previous delta
        for i in reversed(range(len(self.weights) - 1)):
            error = deltas[i + 1].dot(self.weights[i + 1].T)
            deltas[i] = error * self.d_activation_function(self.net[i])

        # Update the weights and the biases
        for i in range(len(self.weights)):
            # Set the previous output to the inputs if there is no previous outputs
            input = np.array(inputs)
            if i != 0:
                input = self.out[i - 1]

            # Update both the weights and the biases
            self.weights[i] += np.atleast_2d(input).T.dot(np.atleast_2d(deltas[i])) * self.learning_rate
            self.biases[i] += deltas[i] * self.learning_rate

    # Trains the neural network using both the feed_forward and the backward_propagation algorithm
    def train(self, inputs, targets, iterations=500000):
        for j in range(iterations):
            i = random.randint(0, len(inputs) - 1)
            self.feed_forward(inputs[i])
            self.backward_propagation(targets[i], inputs[i])

    # Tests the neuronal network using the cost function. Prints out the results
    def test(self, inputs, target):
        loss = []
        out = []
        for i in range(len(inputs)):
            self.feed_forward(inputs[i])
            out.append(self.out[-1].tolist()[0])
            loss.append(np.mean(cost(self.out[-1], target)))
        print("Outputs:", out)
        print("Average Loss:", np.average(loss))

# Calculates the loss of the outputs using the squared loss function
def cost(out, target):
    return 1 / 2 * (np.subtract(target, out) ** 2)

# The derivative of the cost function
def d_cost(out, target):
    return np.subtract(target, out)


# The ReLu activation function and it's derivative as lambdas
relu = lambda x: np.maximum(0, x)
d_relu = lambda x: 1. * (x > 0)

nn = Main([2, 2, 1], activation_function=relu, d_activation_function=d_relu)

nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], 100000)
nn.test([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
