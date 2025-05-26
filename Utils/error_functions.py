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
