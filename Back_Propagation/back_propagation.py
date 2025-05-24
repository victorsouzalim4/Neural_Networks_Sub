from Neuron.neuron import Neuron
import math

def backPropagation(initialLayerWidth, depth, inputs, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):
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
    print(lastLayerLenght)
    outputLayer = [Neuron(lastLayerLenght)]
    layers.append(outputLayer)

    for i, layer in enumerate(layers):
        print(f"\n🧠 Camada {i} ({len(layer)} neurônios)")
        print("-" * 40)

        for j, neuron in enumerate(layer):
            print(f"  🔹 Neurônio {j}: Pesos -> {neuron.weights}")
        print("-" * 40)

        
    
