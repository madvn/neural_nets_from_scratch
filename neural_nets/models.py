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
    def __init__(
        self,
        units_per_layer,
        learning_rate=0.01,
        activation="sigmoid",
        cost="MSE",
        d_cost=None,
    ):
        """ Create FNN based on specifications

        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        learning_rate: (float) learning_rate to update the weights and biases of the networks
        activation: (str or list of strs ,len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid
        cost: (str or lambda) Choose the cost function to use, currently only 'MSE'. Alternatively provide lambda that takes two args, desired_outputs and actual_outputs in that order
        d_cost: (lambda) If cost is a lambda, provide another lambda that also takes same two args to estimate the derivative of cost function with respect to output

        Note: weights are uniform randomly initialized in [-1/sqrt(d), 1/sqrt(d)] where d is the number of inputs a neuron receives
        """
        assert (
            isinstance(units_per_layer, list) and len(units_per_layer) >= 2
        ), "units_per_layer should be a list of len greater than 2 i.e. minimum of 1 input and 1 output layer"
        self.num_layers = len(units_per_layer)
        self.units_per_layer = units_per_layer
        self.learning_rate = learning_rate

        # lambdas for supported activation functions
        self.activation_funcs = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: 2 / (1 + np.exp(-2 * x)) - 1,
            "relu": lambda x: np.maximum(0, x),
        }
        # lambdas for derivatives of supported activation functions
        self.d_activation_funcs = {
            "sigmoid": lambda x: np.asarray(x) * (1 - np.asarray(x)),
            "tanh": lambda x: 1 - (np.asarray(x) ** 2),
            "relu": lambda x: np.asarray([1 if i > 0 else 0 for i in x]),
        }
        # setting activation for each layer
        self._init_activations(activation)

        # lambdas for cost functions - lambda takes desired outputs d, and outputs o
        self.cost_funcs = {
            "MSE": lambda d, o: np.mean(0.5 * (np.asarray(d) - np.asarray(o)) ** 2, 0)
        }

        # lambdas for derivatives of cost functions
        self.d_cost_funcs = {"MSE": lambda d, o: -(d - o)}
        # set cost function
        self._init_cost(cost, d_cost)

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

    def _init_cost(self, cost, d_cost):
        """ Internal function to set cost and derivative of cost for the network """
        # check implementation for cost already exists
        if cost not in self.cost_funcs.keys():
            # check if a valid cost and d_cost has been provided
            throw_away_lambda = lambda: 0
            if (
                isinstance(cost, type(throw_away_lambda))
                and cost.__name__ == throw_away_lambda.__name__
            ) and (
                isinstance(d_cost, type(throw_away_lambda))
                and d_cost.__name__ == throw_away_lambda.__name__
            ):
                self.cost = cost
                self.d_cost = d_cost
            else:
                # no valid cost function has been defined
                raise ValueError(
                    "Choose cost function from {} or provide lambdas for cost and d_cost of __init__".format(
                        self.cost_funcs.keys()
                    )
                )
        else:
            # implementation exists for provided cost function
            self.cost = self.cost_funcs[cost]
            self.d_cost = self.d_cost_funcs[cost]

    def forward(self, inputs):
        """ Forward propogate the given inputs through the network

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
            outputs.append(
                self.activation[l](
                    np.matmul(outputs[-1], self.weights[l])
                )
            )
        return outputs
