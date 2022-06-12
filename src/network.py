import random

class Network: 
    def __init__(self, layer_lengths):
        self.num_layers = len(layer_lengths)
        self.layer_lengths = layer_lengths
        self.weights = [random.randrange(-1, 1) for x in range(len(layer_lengths) - 1)]
        self.biases = [random.randrange(-1, 1) for x in range(len(layer_lengths) - 1)]