##################################################################################
# Feed-forward neural network with gradient descent backprop leanrning
# Works with any number of hidden layers/neurons and supports the following
# activation functions
# 1. sigmoid
# 2. tanh
# 3. ReLU
#
# Madhavun Candadai
# Nov 2018
##################################################################################
import numpy as np
import matplotlib.pyplot as plt


class FNN:
    def __init__(self, units_per_layer, activation="sigmoid"):
        """ Create FNN based on specifications

        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        activation: (str or list of strs ,len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu', 'linear'], default = sigmoid

        Note: weights are uniform randomly initialized in [-1/sqrt(d), 1/sqrt(d)] where d is the number of inputs a neuron receives
        """
        assert (
            isinstance(units_per_layer, list) and len(units_per_layer) >= 2
        ), "units_per_layer should be a list of len greater than 2 i.e. minimum of 1 input and 1 output layer"
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)

        # lambdas for supported activation functions
        self.activation_funcs = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: 2 / (1 + np.exp(-2 * x)) - 1,
            "relu": lambda x: np.maximum(0, x),
            "linear": lambda x: x,
        }
        # lambdas for derivatives of supported activation functions
        self.d_activation_funcs = {
            "sigmoid": lambda x: np.asarray(x) * (1 - np.asarray(x)),
            "tanh": lambda x: 1 - (np.asarray(x) ** 2),
            "relu": lambda x: np.asarray([1 if i > 0 else 0 for i in x]),
            "linear": lambda x: 1,
        }
        # setting activation for each layer
        self._init_activations(activation)

        # assigning intial random weights and biases based on layer size
        self.weights = []
        for l in np.arange(self.num_layers - 1):
            d = -1 / np.sqrt(self.units_per_layer[l] + 1)
            self.weights.append(
                np.random.uniform(
                    low=-d,
                    high=d,
                    size=[self.units_per_layer[l], self.units_per_layer[l + 1]],
                )
            )

    def _init_activations(self, activation):
        """ Internal function to set activation function for each layer"""
        # to verify if implementation exists for provided activation function
        def validate_activation_str(act):
            if act not in self.activation_funcs.keys():
                raise ValueError(
                    "Choose activation from {}".format(self.activation_funcs.keys())
                )

        if isinstance(activation, str):
            # if activation is a string use same activation function for all layers
            activation = activation.strip()
            validate_activation_str(activation)
            self.activation = [self.activation_funcs[activation]] * (
                self.num_layers - 1
            )
            self.d_activation = [self.d_activation_funcs[activation]] * (
                self.num_layers - 1
            )
        else:
            # else activation must be a list of str correspondong to each layer
            assert len(activation) == self.num_layers - 1, (
                "if activation is a list,"
                " len(activation) should be equal to len(units_per_layer)-1, since first layer is input OR"
                " activation should be str to assign same activation to all layers"
            )
            self.activation = []
            # activation is a list
            for act in activation:
                validate_activation_str(act)
                self.activation.append(self.activation_funcs[act])
                self.d_activation.append(self.d_activation_funcs[act])

    def forward(self, inputs):
        """ Forward propagate the given inputs through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network

        RETURNS:
        outputs: (list or matrix) output for all layers for all inputs
        """
        inputs = np.asarray(inputs)
        outputs = [inputs]
        for l in np.arange(self.num_layers - 1):
            if outputs[-1].ndim == 1:
                outputs[-1] = [outputs[-1]]
            outputs.append(self.activation[l](np.matmul(outputs[-1], self.weights[l])))
        return outputs
