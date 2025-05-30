from Perceptron.perceptron import perceptron
from Back_Propagation.back_propagation import backPropagation
from Neuron.neuron import Neuron
from Neural_Network.test_neural_network import testNeuralNetwork
import numpy as np


inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

#weights = [-1.5, -1.2, -1.3]

expectedOutputs = [-1, -1, -1, -1, -1, -1, -1, 1]

#perceptron(inputs, expectedOutputs, 100, "XOR-2bits.gif")

nn = backPropagation(2, 3, inputs, expectedOutputs, 100000, 0.0001, "Tanh-AND-3bits", "tanh", "online")

testNeuralNetwork(nn, inputs, expectedOutputs)