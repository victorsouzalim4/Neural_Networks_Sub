import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D



def saveFrame2D(inputs, expectedOutputs, neuron, epoch, x_min, x_max, y_min, y_max):
    """
    Saves the frame (image) of the current state of the perceptron.

    :param inputs: Input data (already projected if needed)
    :param expectedOutputs: Expected outputs (labels)
    :param neuron: Instance of the neuron
    :param epoch: Current epoch number
    :param x_min, x_max, y_min, y_max: Axis limits for the plot
    :return: Filename of the saved image
    """

    if not os.path.exists("frames"):
        os.makedirs("frames")

    plt.figure(figsize=(8, 6))

    # Plot values according to class
    for i, point in enumerate(inputs):
        if expectedOutputs[i] == 0:
            plt.scatter(point[0], point[1], color='red', label='Class 0' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], color='blue', label='Class 1' if i == 0 else "")

    # Generating the decision boundary
    x_values = np.linspace(x_min, x_max, 100)

    if abs(neuron.weights[1]) < 1e-6:
        x_const = -neuron.biasWeight / (neuron.weights[0] if abs(neuron.weights[0]) > 1e-6 else 1e-6)
        plt.axvline(x=x_const, color='green', label='Decision Boundary')
    else:
        slope = -neuron.weights[0] / neuron.weights[1]
        intercept = -neuron.biasWeight / neuron.weights[1]
        y_values = slope * x_values + intercept
        plt.plot(x_values, y_values, label='Decision Boundary', color='green')

    plt.title(f'Epoch {epoch + 1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    filename = f"frames/frame_{epoch}.png"
    plt.savefig(filename)
    plt.close()

    return filename

def saveFrame3D(inputs, expectedOutputs, neuron, epoch,
                 x_min, x_max, y_min, y_max, z_min, z_max):
    if not os.path.exists("frames"):
        os.makedirs("frames")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points with labels
    for i, point in enumerate(inputs):
        color = 'red' if expectedOutputs[i] == 0 else 'blue'
        label = 'Class 0' if expectedOutputs[i] == 0 else 'Class 1'
        ax.scatter(point[0], point[1], point[2],
                   color=color, label=label if i == 0 or (expectedOutputs[:i].count(expectedOutputs[i]) == 0) else ""
)

    # Create grid for the plane
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                          np.linspace(y_min, y_max, 20))

    if abs(neuron.weights[2]) < 1e-6:
        zz = np.zeros_like(xx)
    else:
        zz = (-neuron.weights[0] * xx - neuron.weights[1] * yy - neuron.biasWeight) / neuron.weights[2]

    # Plot the decision boundary plane
    ax.plot_surface(xx, yy, zz, alpha=0.4, color='green', edgecolor='k', linewidth=0.5)

    # Add vertical dashed lines to the plane for each point
    for i, point in enumerate(inputs):
        x, y, z = point
        if abs(neuron.weights[2]) < 1e-6:
            z_proj = z  # arbitrary when plane is invalid
        else:
            z_proj = (-neuron.weights[0] * x - neuron.weights[1] * y - neuron.biasWeight) / neuron.weights[2]
        ax.plot([x, x], [y, y], [z, z_proj], color='gray', linestyle='dotted')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(f'Epoch {epoch + 1}')

    # Ajustar ângulo da câmera para melhor visualização
    ax.view_init(elev=30, azim=45 + epoch * 5)  # Faz o GIF girar lentamente

    plt.legend()
    filename = f"frames/frame_{epoch}.png"
    plt.savefig(filename)
    plt.close()

    return filename