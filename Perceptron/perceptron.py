from Neuron.neuron import Neuron

def perceptron(inputs, weights, expectedOutputs):

    n = Neuron(weights)
    outputs = []

    while outputs != list(expectedOutputs):
        outputs = [] 

        for i in range(len(inputs)):
            value = n.output(inputs[i])
            output = n.activation(value)

            outputs.append(output)

            n.weightReadjustment(inputs[i], expectedOutputs[i], output)

        print("Epochs outputs: ", outputs)

   

        

        

