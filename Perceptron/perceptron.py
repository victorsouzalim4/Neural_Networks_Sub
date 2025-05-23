
def perceptron(inputs, weights):

    bias = 1
    bias_weight = 0.1
    
    i = 0
    output = 0

    while i != len(inputs):
        output += inputs[i]*weights[i]
        i+=1

    output += bias*bias_weight
    print(output)
