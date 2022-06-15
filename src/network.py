import random
import numpy as np
from helpers import sigmoid, sigmoid_prime
from mnist_loader import load_data_wrapper

class Network: 
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        # is only used in the evaluate method -- update_mini_batch calls backprop which calls its own version
        # of feedforward which is the same in terms of functionality, but also stores values that are necessary 
        # for other backprop steps
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # def SGD(self, training_data=None, epochs=10, mini_batch_size=10, lr=1, test_data=None):
    #     for i in range(epochs):
    #         # random.shuffle(training_data)
    #         mini_batches = [training_data[x:x+mini_batch_size] for x in range(0, len(training_data), mini_batch_size)]
    #         for mini_batch in mini_batches:
    #             self.update_mini_batch(mini_batch, lr)
    #         if test_data:
    #             number_correct = self.evaluate(test_data)
    #             print(f"Epoch {i}: {(number_correct/mini_batch_size)*100}% correct")
    #         else: print(f"Epoch {i} of {epochs}")
    
    def SGD(self, training_data, epochs, mini_batch_size, lr,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                percent_correct = self.evaluate(test_data)/n_test*100
                print(f"Epoch {j}: {percent_correct}% correct")
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, lr):
        # nabla_b = [np.zeros(b.shape) for b in self.biases]
        # nabla_w = [np.zeros(w.shape) for w in self.weights]
        # for guess, actual in mini_batch:
        #     delta_nabla_b, delta_nabla_w = self.backprop(guess, actual)
        #     nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # self.weights = [w-(lr/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        # self.biases = [b-(lr/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(lr/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    # def backprop(self, x, y):
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     activation = x
    #     activations = [x]
    #     zs = []
    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation) + b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)
    #     delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #     for l in range(2, self.num_layers):
    #         z = zs[-l]
    #         sp = sigmoid_prime(z)
    #         delta = np.dot(self.weights[-1+1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    #     return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


# ALTERNATIVE TO RUNNING FROM THE COMMAND LINE
# training_data, validation_data, test_data = load_data_wrapper()
# net = Network([784, 100, 10])
# net.SGD(training_data, 40, 10, 1, test_data)