from Neural_Networks_Sub.Neural_Network.neural_network_gen import neuralNetworkGen
from Neural_Networks_Sub.Neural_Network.forward import passForward
from Neural_Networks_Sub.Utils.plot_Error_Curve import plotErrorCurve
from Neural_Networks_Sub.Utils.activation_functions import getDerivative
from Neural_Networks_Sub.Utils.error_functions import calculateError, calculateOutputLayerError
from Neural_Networks_Sub.Utils.plot_decision_boundary import plotDecisionBoundary


def backPropagation(
    initialLayerWidth,
    depth,
    inputs,
    expectedOutputs,
    maxEpochs=50,
    errorThreshold=0.01,
    fileName="error_curve.png",
    activation="tanh",
    update_mode="online",
    plot_decision_boundary=True
):
    layers = neuralNetworkGen(initialLayerWidth, depth, inputs)
    epoch = 0
    errorMse = float('inf')
    errorHistory = []

    while epoch < maxEpochs and errorMse > errorThreshold:
        outputs = []
        totalError = 0

        batch_weight_updates = [
            [[0] * len(neuron.weights) for neuron in layer]
            for layer in layers
        ]

        for input, expectedOutput in zip(inputs, expectedOutputs):
            outputsPerLayer = passForward(layers, input, activation)
            derivates = []

            output = outputsPerLayer[-1][0]
            outputs.append(output)

            totalError += (expectedOutput - output) ** 2

            for layer in outputsPerLayer:
                derivatesPerLayer = [getDerivative(out, activation) for out in layer]
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
                    layerInputs = input
                else:
                    layerInputs = outputsPerLayer[index - 1]

                for neuronIndex, (neuron, error) in enumerate(zip(layer, errors)):
                    if update_mode == "online":
                        neuron.weightReadjustment(layerInputs, error=error)
                    elif update_mode == "batch":
                        for wIndex in range(len(neuron.weights)):
                            delta = error * layerInputs[wIndex]
                            batch_weight_updates[index][neuronIndex][wIndex] += delta
                    else:
                        raise ValueError("update_mode deve ser 'online' ou 'batch'")

        if update_mode == "batch":
            for layerIndex, layer in enumerate(layers):
                for neuronIndex, neuron in enumerate(layer):
                    for wIndex in range(len(neuron.weights)):
                        avg_delta = batch_weight_updates[layerIndex][neuronIndex][wIndex] / len(inputs)
                        neuron.weights[wIndex] += avg_delta

        errorMse = totalError / len(inputs)
        errorHistory.append(errorMse)

        print(f"Época {epoch + 1}: Erro médio = {errorMse}")

        epoch += 1

    print("Treinamento finalizado.")
    print(outputs)

    plotErrorCurve(errorHistory, fileName)

    if plot_decision_boundary:
        plotDecisionBoundary(
            layers,
            inputs,
            expectedOutputs,
            fileName,
            activation=activation,
            filename=fileName,
            outputDir="Backpropagation_benchmarks/DecisionBoundary"
        )
    

    return layers
