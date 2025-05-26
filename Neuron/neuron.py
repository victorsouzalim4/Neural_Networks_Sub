import numpy as np

class Neuron:

    def __init__(self, nInputs, learningRate = 0.1):
        self.weights = np.random.uniform(-1, 1, nInputs)
        self.bias = 1
        self.biasWeight = np.random.uniform(-1, 1, 1)
        self.learningRate = learningRate

    def netInput(self, inputs):
        return np.dot(self.weights, inputs) + self.bias*self.biasWeight
    
    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))
    
    def stepFunction(self, output):
        return 0 if output <= 0 else 1
    
    def tanh(self, input):
        return np.tanh(input)

    def weightReadjustment(self, inputs, error=None, expectedOutput=None, output=None):

        if error is None:
            if expectedOutput is None or output is None:
                raise ValueError("Se 'error' não for fornecido, 'expectedOutput' e 'output' são obrigatórios.")
            error = expectedOutput - output  # Perceptron

        for i in range(len(self.weights)):
            self.weights[i] += self.learningRate * error * inputs[i]

        self.biasWeight += self.learningRate * error

