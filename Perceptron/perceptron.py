import numpy as np
from Neuron.neuron import Neuron
from Utils.gif_generator import generateGif
from Utils.frame_generator import saveFrame
from sklearn.decomposition import PCA


def perceptron(inputs, weights, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):
    n = Neuron(weights)
    outputs = []

    epochs = 0
    frames = []

    # Check if PCA is needed
    if inputs.shape[1] > 2:
        print("Input has more than 2 features. Applying PCA for visualization...")
        pca = PCA(n_components=2)
        inputs_projected = pca.fit_transform(inputs)
    else:
        inputs_projected = inputs

    # Setting limits for the graph
    margin = 0.5
    x_min, x_max = np.min(inputs_projected[:, 0]) - margin, np.max(inputs_projected[:, 0]) + margin
    y_min, y_max = np.min(inputs_projected[:, 1]) - margin, np.max(inputs_projected[:, 1]) + margin

    while outputs != list(expectedOutputs) and epochs < max_epochs:
        outputs = []

        frame = saveFrame(
            inputs_projected, expectedOutputs, n, epochs,
            x_min, x_max, y_min, y_max
        )
        frames.append(frame)

        for i in range(len(inputs)):
            value = n.output(inputs[i])  # Training happens in the original input space
            output = n.activation(value)

            outputs.append(output)

            n.weightReadjustment(inputs[i], expectedOutputs[i], output)

        print(f"Epoch {epochs}: Outputs -> {outputs}")
        epochs += 1

    generateGif(frames, output_filename=gif_name, fps=5)
