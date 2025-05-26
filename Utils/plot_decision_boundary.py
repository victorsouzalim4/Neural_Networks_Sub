import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from Neural_Network.forward import passForward


def plotDecisionBoundary(
    layers,
    inputs,
    expectedOutputs,
    title,
    activation="tanh",
    filename="decision_boundary.png",
    outputDir="Backpropagation_benchmarks/DecisionBoundary"
):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    num_features = len(inputs[0])
    if num_features not in [2, 3]:
        raise ValueError("Apenas redes com 2 ou 3 entradas são suportadas para plotDecisionBoundary.")

    inputs = np.array(inputs)
    expectedOutputs = np.array(expectedOutputs)

    if num_features == 2:
        # Plot para 2 entradas com fronteira de decisão

        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 500),
            np.linspace(y_min, y_max, 500)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = []
        for point in grid_points:
            output = passForward(layers, point, activation)[-1][0]
            Z.append(output)

        Z = np.array(Z).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 200), cmap=plt.cm.RdBu, alpha=0.6)
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        plt.scatter(
            inputs[:, 0],
            inputs[:, 1],
            c=expectedOutputs,
            cmap=plt.cm.RdBu,
            edgecolor='k',
            s=80
        )

        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)

        filepath = os.path.join(outputDir, filename)
        plt.savefig(filepath)
        plt.close()

        print(f"Gráfico salvo em: {filepath}")

    elif num_features == 3:
        # Plot 3D para dados com 3 entradas

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            c=expectedOutputs,
            cmap=plt.cm.RdBu,
            edgecolor='k',
            s=80
        )

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.set_title('Distribuição no Espaço - 3 Entradas')

        ax.view_init(elev=30, azim=135)  # Ângulo da câmera ajustável

        filepath = os.path.join(outputDir, filename)
        plt.savefig(filepath)
        plt.close()

        print(f"Gráfico 3D salvo em: {filepath}")
