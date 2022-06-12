import random
import numpy as np

class Network: 
    def __init__(self, layer_lengths):
        self.num_layers = len(layer_lengths)
        self.layer_lengths = layer_lengths
        self.biases = [np.random.rand(x, 1) for x in layer_lengths[1:]]
        self.weights = [np.random.rand(x, y) for x, y in zip(layer_lengths[:-1], layer_lengths[1:])]