import _thread
import threading

from Python import Main


class Thread (threading.Thread):
    def __init__(self, nn: Main, inputs, targets, iterations):
        threading.Thread.__init__(self)
        self.nn = nn
        self.inputs = inputs
        self.targets = targets
        self.weights = []
        self.biases = []
        self.iterations = iterations

    def run(self):
        self.nn.feed_forward(self.inputs, self.targets, self.iterations)
        self.nn.back_propagation(self.targets, self.inputs)
        print(self.nn.out[-1])

        self.weights = self.nn.weights
        self.biases = self.nn.biases

