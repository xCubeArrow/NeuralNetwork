import copy
import json
from multiprocessing.pool import ThreadPool
import multiprocessing
import random
import threading

import numpy as np


# The default activation function that the neural network uses, as lambdas


def thread_train(neural_network, inputs, targets, iterations):
    nn = copy.copy(neural_network)
    nn.train(inputs, targets, round(iterations))
    return [nn.weights, nn.biases]


def sigmoid(x): return x / (1 + abs(x))


def d_sigmoid(x): return 1 / (1 + pow(abs(x), 2))


# The ReLu activation function and it's derivative as lambdas
def relu(x): return np.maximum(0, x)


def d_relu(x): return 1. * (x > 0)


activation_dir = {
    "sigmoid": sigmoid,
    "d_sigmoid": d_sigmoid,

    "relu": relu,
    "d_relu": d_relu
}


class Main:
    # Initialize the neural network with all of the required variables
    def __init__(self, layers, learning_rate=.1, activation_function=sigmoid, d_activation_function=d_sigmoid,
                 weights=None, biases=None):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function
        self.cost = 99

        if weights is not None:
            self.weights = weights
            self.biases = biases
        else:
            # Make both the weights and the biases random
            self.weights = []
            self.biases = []
            for i in range(1, len(layers)):
                self.weights.append(np.random.rand(layers[i - 1], layers[i]))
                self.biases.append(np.random.rand(1, layers[i]))
        self.net = [None] * (len(self.weights))
        self.out = [None] * (len(self.weights))

    @staticmethod
    def from_file(location: str):
        with open(location) as file:
            json_nn = json.loads(file.read())
            learning_rate = json_nn["learning_rate"]
            activation_function = activation_dir[json_nn["activation_function"]]
            d_activation_function = activation_dir[json_nn["d_activation_function"]]

            weights = []
            biases = []
            for obj in json_nn["layers"]:
                weights.append(np.asarray(obj["weights"]))
                biases.append(np.asarray(obj["biases"]))
            file.close()

            return Main(None, learning_rate=learning_rate, activation_function=activation_function,
                        d_activation_function=d_activation_function, weights=weights, biases=biases)

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
        self.cost = np.average(loss)

    def save_nn(self, location):
        file = open(location, "w")

        starting = [
            '"learning_rate": ' + str(self.learning_rate),
            '"activation_function": "' + self.activation_function.__name__ + '"',
            '"d_activation_function": "' + self.d_activation_function.__name__ + '"'
        ]

        result = []
        for i in range(len(self.weights)):
            weight = '"weights": ' + json.dumps(self.weights[i].tolist())
            biases = '"biases": ' + json.dumps(self.biases[i].tolist())
            result.append(json.loads("{" + weight + ", " + biases + "}"))

        file.write("{" + ", ".join(starting) + ', "layers":' + json.dumps(result) + "}")
        file.close()


# Calculates the loss of the outputs using the squared loss function
def cost(out, target):
    return 1 / 2 * (np.subtract(target, out) ** 2)


# The derivative of the cost function
def d_cost(out, target):
    return np.subtract(target, out)


nn = Main(layers=[2, 2, 1])
nn_inputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
nn_targets = [[0], [1], [1], [0]]
# with open("divorce.csv", "r") as file:
#    data = file.read()
#    for line in data.split("\n"):
#        inputs.append(np.array(line.split(";")[:-1]).astype(np.double).tolist())
#        targets.append(np.array(line.split(";")[-1]).astype(np.double).tolist())

iteration = 1
nn.train(nn_inputs, nn_targets, 100000)
nn.test(nn_inputs, nn_targets)
print("Iteration", iteration)
iteration += 1
nn.save_nn("nn.json")
