from Neuron.neuron import Neuron

def perceptron(inputs, weights, expectedOutputs):

    n = Neuron(weights)

    value = n.output(inputs[0])
    output = n.activation(value)
    print(value)
    n.weightReadjustment(inputs[0], expectedOutputs[0], output)

    value = n.output(inputs[0])
    print(value)

