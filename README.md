# Implementations of Neural Network models and training algorithms from scratch [WIP]

## Models and Algos
- [X] Feedforward network of arbitrary size and activation functions

- [X] Backprop with arbitrary cost function

- [X] Feedback alignment based training for FNNs [1]

- [X] Recurrent network with arbitrary size and activtion functions

- [ ] Backprop through time [WIP]

- [ ] RNN traninng with feedback alignment [2]


### References
[1] Lillicrap, Timothy P., et al. Random synaptic feedback weights support error backpropagation for deep learning. Nature communications 7 (2016): 13276

[2] Murray, J. M. (2018). Local online learning in recurrent networks with random feedback. BioRxiv, 458570.

<!-- ## Feed-forward neural network (FNN - neuralNets/fnn.py/FNN)

#### Create object of FNN with __init__ as follows

        def __init__(self, units_per_layer, learning_rate=0.01, activation='sigmoid', cost='MSE',d_cost=None)

ARGS

**units_per_layer:** (list, len>=2) Number of neurons in each layer including input, hidden and output

**learning_rate:** (float) learning_rate to update the weights and biases of the networks

**activation:** (str or list of strs with len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid

**cost:** (str or lambda) Choose the cost function to use, currently only 'MSE'. Alternatively provide lambda that takes two args, desired_outputs and actual_outputs in that order

**d_cost:** (lambda) If cost is a lambda, provide another lambda that also takes same two args to estimate the derivative of cost function with respect to output


#### Train
Call training_step several times with inputs and desired outputs - for example

        nn = FNN([2,3,2,2],activation='tanh')
        inputs = [[-1,-1],[-1,1],[1,-1],[1,1]]
        outputs = [[-1,-1],[1,1],[1,1],[1,-1]] # 2 outputs - OR and XOR logic gates
        errs = []
        for _ in range(10000):
            error = nn.training_step(inputs, outputs)
            errs.append(error[1])

-->
