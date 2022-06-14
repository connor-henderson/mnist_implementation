import random
import numpy as np
from helpers import sigmoid, sigmoid_prime

class Network: 
    def __init__(self, layer_lengths):
        self.num_layers = len(layer_lengths)
        self.layer_lengths = layer_lengths
        self.biases = [np.random.rand(x, 1) for x in layer_lengths[1:]]
        self.weights = [np.random.rand(x, y) for x, y in zip(layer_lengths[:-1], layer_lengths[1:])]
    
    def feedforward(self, a):
        # is only used in the evaluate method -- update_mini_batch calls backprop which calls its own version
        # of feedforward which is the same in terms of functionality, but also stores values that are necessary 
        # for other backprop steps
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, lr=1, mini_batch_size=10, epochs=10, training_data=None, test_data=None):
        training_data_length = len(training_data)
        for i in range(1, epochs + 1):
            shuffled_training_data = random.shuffle(training_data)
            mini_batches = [shuffled_training_data[x:x+mini_batch_size] for x in range(0, training_data_length, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                number_correct = self.evaluate(test_data)
                print(f"Epoch {i}: {(number_correct/mini_batch_size)*100}% correct")
            else: print(f"Epoch {i} of {epochs}")

    def update_mini_batch(self, mini_batch, lr):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for guess, actual in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(guess, actual)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(lr/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-1+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        number_correct = 0
        for x, y in test_data:
            a = self.feedforward(x)
            if a == y: number_correct += 1
        return number_correct



# TEST AREA

# import random
# rand_output = np.random.rand(10, 1)
# rand_sigmoid = sigmoid(rand_output)
# print(rand_output, '\n', rand_sigmoid)