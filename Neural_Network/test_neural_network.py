from Neural_Network.forward import passForward

def testNeuralNetwork(layers, test_inputs, test_expected_outputs=None, activation="tanh", threshold=0.0):
    predictions = []
    correct = 0

    for i, input_vector in enumerate(test_inputs):
        outputs = passForward(layers, input_vector, activation)
        final_output = outputs[-1][0]

        # Classificação baseada em limiar
        predicted_class = 1 if final_output >= threshold else -1
        predictions.append(predicted_class)

        if test_expected_outputs:
            expected = test_expected_outputs[i]
            print(f"Input: {input_vector}, Expected: {expected}, Predicted: {predicted_class}")
            if predicted_class == expected:
                correct += 1
        else:
            print(f"Input: {input_vector}, Predicted: {predicted_class}")

    if test_expected_outputs:
        accuracy = correct / len(test_inputs)
        print(f"\nAcurácia: {accuracy * 100:.2f}%")

    return predictions
