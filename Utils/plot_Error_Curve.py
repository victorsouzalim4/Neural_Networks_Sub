import matplotlib.pyplot as plt

def plotErrorCurve(errorHistory):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(errorHistory) + 1), errorHistory, marker='o')
    plt.title('Curva de Erro Durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Médio')
    plt.grid(True)
    plt.show()
