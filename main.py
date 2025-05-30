from Neural_Networks_Sub.Back_Propagation.back_propagation import backPropagation
from Neural_Networks_Sub.Neuron.neuron import Neuron
from Neural_Networks_Sub.Neural_Network.test_neural_network import testNeuralNetwork
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

inputs = np.array([
    [0, 0.1, 0.05],
    [0, 0.4, 1],
    [0.2, 1, 0],
    [0, 0.95, 1],
    [1, 0.3, 0.01],
    [1, 0.1, 1],
    [0.99, 1, 0],
    [0.97, 0.92, 0.98],
])

testNeuralNetwork(nn, inputs, expectedOutputs)