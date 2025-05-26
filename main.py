from Perceptron.perceptron import perceptron
from Back_Propagation.back_propagation import backPropagation
from Neuron.neuron import Neuron
import numpy as np


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

weights = [0.5, 0.2, 0.3]

expectedOutputs = [0, 0, 0, 1]

#perceptron(inputs, expectedOutputs, 100, "XOR-2bits.gif")

backPropagation(2, 3, inputs, expectedOutputs, 10000, 0.01, "Tanh-AND-2bits", "sigmoid")