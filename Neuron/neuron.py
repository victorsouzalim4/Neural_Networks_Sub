import numpy as np

class Neuron:

    def __init__(self, weights, learningRate = 0.01):
        self.weights = weights
        self.bias = 1
        self.biasWeight = 0.1
        self.learningRate = learningRate

    def output(self, inputs):
        return np.dot(self.weights, inputs) + self.bias*self.biasWeight
    
    def activation(self, output):
        return 0 if output <= 0 else 1
    
    def weightReadjustment(self, inputs, expectedOutput, output):
        error = expectedOutput - output

        for i in range(len(self.weights)):
            self.weights[i] += self.learningRate * error * inputs[i]

        self.biasWeight += self.learningRate * error
