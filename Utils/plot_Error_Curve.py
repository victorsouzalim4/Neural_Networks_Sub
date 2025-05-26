import matplotlib.pyplot as plt
import os

def plotErrorCurve(errorHistory, filename="error_curve.png"):
    # Cria o diretório se não existir
    outputDir = os.path.join("Backpropagation_benchmarks", "LinePlot")
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Cria o gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(errorHistory) + 1), errorHistory, marker='o')
    plt.title(filename)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Médio')
    plt.grid(True)

    # Salva o gráfico no diretório
    filepath = os.path.join(outputDir, filename)
    plt.savefig(filepath)

    print(f"Gráfico salvo em: {filepath}")

    plt.show()
