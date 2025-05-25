from Neuron.neuron import Neuron
from Utils.neural_network_gen import neuralNetworkGen
import math

def backPropagation(initialLayerWidth, depth, inputs, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):

    layers = neuralNetworkGen(initialLayerWidth, depth, inputs)
    epoch = 0

    while epoch < max_epochs:
        outputs = []

        for input, expectedOutput in zip(inputs, expectedOutputs):
            outputsPerLayer = passFoward(layers, input)
            derivates = []

            outputs.append(outputsPerLayer[-1])

            for layer in outputsPerLayer:
                derivatesPerLayer = [sigmoidDerivative(output) for output in layer]
                derivates.append(derivatesPerLayer)

            outputLayerErrors = calculateOutputLayerError(
                expectedOutput,
                derivates[-1],
                outputsPerLayer[-1]
            )

            nextLayerErrors = outputLayerErrors
            nextLayer = layers[-1]

            errorsPerLayer = [outputLayerErrors]

            for i in range(len(layers) - 1):
                layerErrors = calculateError(
                    layers[-i - 2],
                    nextLayer,
                    nextLayerErrors,
                    outputsPerLayer[-i - 2],
                    derivates[-i - 2]
                )
                errorsPerLayer.append(layerErrors)
                nextLayerErrors = layerErrors
                nextLayer = layers[-i - 2]

            errorsPerLayer.reverse()

            for index, (layer, errors) in enumerate(zip(layers, errorsPerLayer)):
                if index == 0:
                    layer_inputs = input
                else:
                    layer_inputs = outputsPerLayer[index - 1]

                for neuron, error in zip(layer, errors):
                    neuron.weightReadjustment(layer_inputs, error=error)

        epoch += 1
        print(outputs)



def passFoward(layers, input):
    
    data = input
    outputsPerLayer = []
    # i = 0
    for layer in layers:
        layerOutputs = []

        for neuron in layer:
            linearOutput = neuron.netInput(data)
            layerOutputs.append(neuron.sigmoid(linearOutput))

        data = layerOutputs
        outputsPerLayer.append(data)
        # print(f"\n Camada {i}")
        # print(data)
        # i+=1
        # print(layerOutputs)

    return outputsPerLayer
    
def sigmoidDerivative(activation):
    return activation * (1 - activation)

def calculateError(layer, nextLayer, nextLayerErrors, layerOutput, derivates):
    """
    Calcula o erro (delta) para cada neurônio de uma camada oculta.

    Args:
        layer (list): Lista dos neurônios da camada atual.
        nextLayer (list): Lista dos neurônios da camada seguinte.
        nextLayerErrors (list): Lista dos deltas (erros) da camada seguinte.
        layerOutput (list): Saídas (ativadas) da camada atual.
        derivates (list): Derivadas da função de ativação da camada atual.

    Returns:
        list: Lista de erros (deltas) para cada neurônio da camada atual.
    """
    errors = []

    for i, neuron in enumerate(layer):
        error_sum = 0

        for j, nextNeuron in enumerate(nextLayer):
            # Pega o peso que conecta este neurônio atual (i) ao neurônio da camada seguinte (j)
            weight = nextNeuron.weights[i]  # ✔️ Aqui está o ponto chave
            error_sum += weight * nextLayerErrors[j]

        # Calcula delta (erro) para este neurônio
        error = derivates[i] * error_sum
        errors.append(error)

    return errors

def calculateOutputLayerError(expectedOutput, derivates, output):

    outputLayerErrors = []
    for i in range(len(derivates)):
        outputLayerErrors.append((expectedOutput - output[i]) * derivates[i])

    return outputLayerErrors

    