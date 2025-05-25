from Neuron.neuron import Neuron
from Utils.neural_network_gen import neuralNetworkGen
import math

def backPropagation(initialLayerWidth, depth, inputs, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):

    layers = neuralNetworkGen(initialLayerWidth, depth, inputs)

    


    # for i, layer in enumerate(layers):
    #     print(f"\n🧠 Camada {i} ({len(layer)} neurônios)")
    #     print("-" * 40)

    #     for j, neuron in enumerate(layer):
    #         print(f"  🔹 Neurônio {j}: Pesos -> {neuron.weights}")
    #     print("-" * 40)

        
    
