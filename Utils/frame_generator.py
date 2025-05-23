import matplotlib.pyplot as plt
import numpy as np
import os


def saveFrame(inputs, expectedOutputs, neuron, epoch, x_min, x_max, y_min, y_max):
    """
    Salva o frame (imagem) da situação atual do perceptron.

    :param inputs: Dados de entrada (numpy array)
    :param expectedOutputs: Saídas esperadas
    :param neuron: Instância do neurônio
    :param epoch: Número da época atual
    :param x_min, x_max, y_min, y_max: Limites dos eixos
    :return: Nome do arquivo salvo
    """
    if not os.path.exists("frames"):
        os.makedirs("frames")

    plt.figure(figsize=(8, 6))

    # Plotando os pontos conforme as classes
    for i, point in enumerate(inputs):
        if expectedOutputs[i] == 0:
            plt.scatter(point[0], point[1], color='red', label='Classe 0' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], color='blue', label='Classe 1' if i == 0 else "")

    # Gerando a fronteira de decisão
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

    # Fixando os limites
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    filename = f"frames/frame_{epoch}.png"
    plt.savefig(filename)
    plt.close()

    return filename
