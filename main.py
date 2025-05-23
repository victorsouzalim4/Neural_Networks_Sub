from Perceptron.perceptron import perceptron
from Neuron.neuron import Neuron
import numpy as np

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

weights = [0.5, 0.2]

expectedOutputs = [0, 0, 0, 1]

perceptron(inputs, weights, expectedOutputs, 100, "AND.gif")