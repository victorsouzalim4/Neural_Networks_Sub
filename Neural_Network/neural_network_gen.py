from Neural_Networks_Sub.Neuron.neuron import Neuron
import math

def neuralNetworkGen(initialLayerWidth, depth, inputs):
    intermediaryLayersWidth = math.ceil((initialLayerWidth + 1) / 2)

    layers = [] #list of neuron lists
    layer = []  #neuron list


    #input layer
    for i in range(initialLayerWidth):
        layer.append(Neuron(len(inputs[0])))

    lastLayerLenght = len(layer)
    layers.append(layer)
    
    # hidden layers
    for i in range(depth - 2):
        layer = []

        for j in range(intermediaryLayersWidth):
            layer.append(Neuron(lastLayerLenght))

        lastLayerLenght = len(layer)
        layers.append(layer)
    
    #output layer
    outputLayer = [Neuron(lastLayerLenght)]
    layers.append(outputLayer)

    return layers

        
    
