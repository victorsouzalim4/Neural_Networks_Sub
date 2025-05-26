def getDerivative(activation, activationFunction):
    if activationFunction == "sigmoid":
        return sigmoidDerivative(activation)
    elif activationFunction == "tanh":
        return tanhDerivative(activation)
    else:
        raise ValueError(f"Derivada para ativação '{activationFunction}' não implementada.")


def sigmoidDerivative(activation):
    return activation * (1 - activation)


def tanhDerivative(activation):
    return 1 - activation ** 2