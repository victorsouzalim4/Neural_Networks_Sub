from Neuron.neuron import Neuron
from Utils.neural_network_gen import neuralNetworkGen
import math

def backPropagation(initialLayerWidth, depth, inputs, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):

    layers = neuralNetworkGen(initialLayerWidth, depth, inputs)

    outputsPerLayer = passFoward(layers, inputs)

    derivates = []

    for layer in outputsPerLayer:
        derivatesPerLayer = []
        for output in layer:
            derivatesPerLayer.append(sigmoidDerivative(output))
        derivates.append(derivatesPerLayer)

    
    #print(derivates)

    #print(outputsPerLayer)

    outputLayerErrors = calculateOutputLayerError(expectedOutputs[0], derivates[len(derivates) - 1], outputsPerLayer[len(outputsPerLayer) - 1])

    nextLayerErrors = outputLayerErrors
    nextLayer = layers[len(layers) - 1]

    errorsPerLayer = []
    errorsPerLayer.append(outputLayerErrors)

    for i in range(len(layers) - 1):  # -1 porque não calcula erro para a camada de saída
        layerErrors = calculateError(
            layers[len(layers) - i - 2],                     # 🔥 camada atual (de trás pra frente)
            nextLayer,                                       # 🔥 camada seguinte
            nextLayerErrors,                                 # 🔥 erros da camada seguinte
            outputsPerLayer[len(layers) - i - 2],            # 🔥 outputs da camada atual
            derivates[len(derivates) - i - 2]                # 🔥 derivadas da camada atual
        )
        errorsPerLayer.append(layerErrors)
        nextLayerErrors = layerErrors
        nextLayer = layers[len(layers) - i - 2]
    
    print(errorsPerLayer)

    errorsPerLayer.reverse()


    for layer, errors in zip(layers, errorsPerLayer):
        for neuron, error in zip(layer, errors):
            neuron.weightReadjustment(inputs[0], error)


    # for i, layer in enumerate(layers):
    #     print(f"\n🧠 Camada {i} ({len(layer)} neurônios)")
    #     print("-" * 40)

    #     for j, neuron in enumerate(layer):
    #         print(f"  🔹 Neurônio {j}: Pesos -> {neuron.weights}")
    #     print("-" * 40)


def passFoward(layers, inputs):
    
    data = inputs[0]
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

    