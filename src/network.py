import numpy as np
from helpers import sigmoid

class Network: 
    def __init__(self, layer_lengths):
        self.num_layers = len(layer_lengths)
        self.layer_lengths = layer_lengths
        self.biases = [np.random.rand(x, 1) for x in layer_lengths[1:]]
        self.weights = [np.random.rand(x, y) for x, y in zip(layer_lengths[:-1], layer_lengths[1:])]
    
    def feedforward(self, a):
        # this feeds an activation (a) through the network,
        # with the intial activation being the input to the network
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a









# TEST AREA

import random
rand_output = np.random.rand(10, 1)
rand_sigmoid = sigmoid(rand_output)
print(rand_output, '\n', rand_sigmoid)