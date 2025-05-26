from Neuron.neuron import Neuron
from Utils.neural_network_gen import neuralNetworkGen
from Utils.plot_Error_Curve import plotErrorCurve
import math


def backPropagation(initialLayerWidth, depth, inputs, expectedOutputs, maxEpochs=50, errorThreshold=0.01, fileName="grafico", activation="tanh"):
    layers = neuralNetworkGen(initialLayerWidth, depth, inputs)
    epoch = 0
    errorMedio = float('inf')
    errorHistory = []

    while epoch < maxEpochs and errorMedio > errorThreshold:
        outputs = []
        totalError = 0

        for input, expectedOutput in zip(inputs, expectedOutputs):
            outputsPerLayer = passForward(layers, input, activation)
            derivates = []

            output = outputsPerLayer[-1][0]  # Saída da última camada
            outputs.append(output)

            totalError += abs(expectedOutput - output)

            # 🔥 Cálculo das derivadas
            for layer in outputsPerLayer:
                derivatesPerLayer = [getDerivative(out, activation) for out in layer]
                derivates.append(derivatesPerLayer)

            # 🔥 Cálculo dos erros da camada de saída
            outputLayerErrors = calculateOutputLayerError(
                expectedOutput,
                derivates[-1],
                outputsPerLayer[-1]
            )

            nextLayerErrors = outputLayerErrors
            nextLayer = layers[-1]

            errorsPerLayer = [outputLayerErrors]

            # 🔥 Cálculo dos erros para camadas ocultas
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

            # 🔥 Ajuste dos pesos
            for index, (layer, errors) in enumerate(zip(layers, errorsPerLayer)):
                if index == 0:
                    layerInputs = input
                else:
                    layerInputs = outputsPerLayer[index - 1]

                for neuron, error in zip(layer, errors):
                    neuron.weightReadjustment(layerInputs, error=error)

        errorMedio = totalError / len(inputs)
        errorHistory.append(errorMedio)

        print(f"Época {epoch + 1}: Erro médio = {errorMedio}")
        epoch += 1

    print("Treinamento finalizado.")
    print(outputs)

    plotErrorCurve(errorHistory, fileName)


# 🔥 Pass Forward parametrizado
def passForward(layers, input, activation="tanh"):
    data = input
    outputsPerLayer = []

    for layer in layers:
        layerOutputs = []

        for neuron in layer:
            linearOutput = neuron.netInput(data)

            if activation == "sigmoid":
                layerOutputs.append(neuron.sigmoid(linearOutput))
            elif activation == "tanh":
                layerOutputs.append(neuron.tanh(linearOutput))
            else:
                raise ValueError(f"Função de ativação '{activation}' não suportada.")

        data = layerOutputs
        outputsPerLayer.append(layerOutputs)

    return outputsPerLayer


# 🔥 Funções de derivada parametrizadas
def getDerivative(activation, activationFunction):
    if activationFunction == "sigmoid":
        return sigmoidDerivative(activation)
    elif activationFunction == "tanh":
        return tanhDerivative(activation)
    else:
        raise ValueError(f"Derivada para ativação '{activationFunction}' não implementada.")


def sigmoidDerivative(activation):
    return activation * (1 - activation)


def tanhDerivative(activation):
    return 1 - activation ** 2


def calculateError(layer, nextLayer, nextLayerErrors, layerOutput, derivates):
    errors = []

    for i, neuron in enumerate(layer):
        errorSum = 0

        for j, nextNeuron in enumerate(nextLayer):
            weight = nextNeuron.weights[i]
            errorSum += weight * nextLayerErrors[j]

        error = derivates[i] * errorSum
        errors.append(error)

    return errors


def calculateOutputLayerError(expectedOutput, derivates, output):
    outputLayerErrors = []
    for i in range(len(derivates)):
        outputLayerErrors.append((expectedOutput - output[i]) * derivates[i])
    return outputLayerErrors
