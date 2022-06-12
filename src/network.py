import random
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

    # implement SGD
    def SGD(self, lr=1, mini_batch_size=10, epochs=10, training_data=None):
        training_data_length = len(training_data)
        for i in range(1, epochs + 1):
            shuffled_training_data = random.shuffle(training_data)
            mini_batches = [shuffled_training_data[:x] for x in range(0, training_data_length, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            print(f"Epoch {i} of {epochs}")

    def backprop(self):
        pass

    def update_mini_batch(self):
        pass







# TEST AREA

# import random
# rand_output = np.random.rand(10, 1)
# rand_sigmoid = sigmoid(rand_output)
# print(rand_output, '\n', rand_sigmoid)