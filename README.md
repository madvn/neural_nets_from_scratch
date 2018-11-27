# neuralNets
Implementation of generic backprop from scratch


#### Feed-forward neural network (FNN - neuralNets/fnn.py/FNN)

Create object of FNN with the following args

        def __init__(self, units_per_layer, learning_rate=0.01, activation='sigmoid', cost='MSE',d_cost=None)

            units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
            learning_rate: (float) learning_rate to update the weights and biases of the networks
            activation: (str or list of strs with len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid


Call training_step several times with inputs and desired outputs - for example

        nn = FNN([2,3,2,2],activation='tanh')
        inputs = [[-1,-1],[-1,1],[1,-1],[1,1]]
        outputs = [[-1,-1],[1,1],[1,1],[1,-1]] # 2 outputs - OR and XOR logic gates
        errs = []
        for _ in range(10000):
            error = nn.training_step(inputs, outputs)
            errs.append(error[1])
