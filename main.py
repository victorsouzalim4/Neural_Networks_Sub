from Perceptron.perceptron import perceptron
from Neuron.neuron import Neuron

inputs = [1, 0]
weights = [0.5, 0.2]

n = Neuron(weights)

output = n.output(inputs)

print(n.activation(output))