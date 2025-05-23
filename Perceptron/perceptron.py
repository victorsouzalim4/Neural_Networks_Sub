import numpy as np
from Neuron.neuron import Neuron
from Utils.gif_generator import generateGif
from Utils.frame_generator import saveFrame


def perceptron(inputs, weights, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):
    n = Neuron(weights)
    outputs = []

    epochs = 0
    frames = []

    # setting limits for graph
    margin = 0.5
    x_min, x_max = np.min(inputs[:, 0]) - margin, np.max(inputs[:, 0]) + margin
    y_min, y_max = np.min(inputs[:, 1]) - margin, np.max(inputs[:, 1]) + margin

    while outputs != list(expectedOutputs) and epochs < max_epochs:
        outputs = []

        if inputs.shape[1] == 2:
            frame = saveFrame(inputs, expectedOutputs, n, epochs, x_min, x_max, y_min, y_max)
            frames.append(frame)
        else:
            print("Decision boundary can not be plotted, because it requires exactly 2 inputs.")

        for i in range(len(inputs)):
            value = n.output(inputs[i])
            output = n.activation(value)

            outputs.append(output)

            n.weightReadjustment(inputs[i], expectedOutputs[i], output)

        print(f"Época {epochs}: Outputs -> {outputs}")
        epochs += 1

    generateGif(frames, output_filename=gif_name, fps=5)
