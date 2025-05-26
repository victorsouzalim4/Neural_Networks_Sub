import numpy as np
from Neuron.neuron import Neuron
from Utils.gif_generator import generateGif
from Utils.frame_generator import saveFrame2D, saveFrame3D
from sklearn.decomposition import PCA


def perceptron(inputs, expectedOutputs, max_epochs=50, gif_name="perceptron_training.gif"):
    n = Neuron(len(inputs[0]))
    outputs = []

    epochs = 0
    frames = []

    num_features = inputs.shape[1]

    # Check visualization mode
    if num_features == 2:
        visualization_mode = "2D"
        inputs_projected = inputs
    elif num_features == 3:
        visualization_mode = "3D"
        inputs_projected = inputs
    else:
        print("Input has more than 3 features. Visualization is disabled.")
        inputs_projected = None
        visualization_mode = None

    # Setting limits for the graph (only if visualization)
    if visualization_mode in ["2D", "3D"]:
        margin = 0.5
        x_min, x_max = np.min(inputs[:, 0]) - margin, np.max(inputs[:, 0]) + margin
        y_min, y_max = np.min(inputs[:, 1]) - margin, np.max(inputs[:, 1]) + margin
        if num_features == 3:
            z_min, z_max = np.min(inputs[:, 2]) - margin, np.max(inputs[:, 2]) + margin

    while outputs != list(expectedOutputs) and epochs < max_epochs:
        outputs = []

        if visualization_mode == "2D":
            frame = saveFrame2D(
                inputs_projected, expectedOutputs, n, epochs,
                x_min, x_max, y_min, y_max
            )
            frames.append(frame)

        if visualization_mode == "3D":
            frame = saveFrame3D(
                inputs_projected, expectedOutputs, n, epochs,
                x_min, x_max, y_min, y_max, z_min, z_max
            )
            frames.append(frame)

        for i in range(len(inputs)):
            linearOutput = n.netInput(inputs[i])
            output = n.stepFunction(linearOutput)

            outputs.append(output)
            n.weightReadjustment(inputs[i], None, expectedOutputs[i], output)

        print(f"Epoch {epochs}: Outputs -> {outputs}")
        epochs += 1

    if frames:
        generateGif(frames, output_filename=gif_name, fps=2)
    else:
        print("No frames generated. Training completed without visualization.")
