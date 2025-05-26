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
