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


def drelu_du(u):
    """derivative of relu w.r.t. input"""
    o = np.zeros_like(u)
    o[u>0] = 1
    return o


class NeuralNet:
    def __init__(self, units_per_layer, activation="sigmoid", cost="MSE", d_cost=None):
        """ Create NeuralNet based on specifications

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
            "relu": drelu_du,
            "linear": lambda x: 1,
        }

        # setting activation for each layer
        self._init_activations(activation)

        # assigning intial random weights and biases based on layer size
        self.weights = []
        self.biases = []
        for l in np.arange(self.num_layers - 1):
            d = 1 / np.sqrt(self.units_per_layer[l] + 1)
            self.weights.append(
                np.random.uniform(
                    low=-d,
                    high=d,
                    size=[self.units_per_layer[l], self.units_per_layer[l + 1]],
                )
            )
            self.biases.append(
                np.random.uniform(
                    low=-d,
                    high=d,
                    size=[1,self.units_per_layer[l+1]],
                )
            )

        # lambdas for cost functions - lambda takes desired outputs d, and outputs o
        self.cost_funcs = {
            "MSE": lambda d, o: np.mean(0.5 * (np.asarray(d) - np.asarray(o)) ** 2, 0)
        }

        # lambdas for derivatives of cost functions
        self.d_cost_funcs = {"MSE": lambda d, o: -(d - o)}

        # set cost function
        self._init_cost(cost, d_cost)

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
            self.d_activation = []
            # activation is a list
            for act in activation:
                validate_activation_str(act)
                self.activation.append(self.activation_funcs[act])
                self.d_activation.append(self.d_activation_funcs[act])

    def setParams(self,params):
        self.weights = []
        start = 0
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l]*self.units_per_layer[l+1]
            self.weights.append(params[start:end].reshape(self.units_per_layer[l],self.units_per_layer[l+1]))
            start = end
        self.biases = []
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l+1]
            self.biases.append(params[start:end].reshape(1,self.units_per_layer[l+1]))
            start = end

    def getParams(self):
        params = np.zeros(self.paramsize)
        start = 0
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l]*self.units_per_layer[l+1]
            params[start:end] = self.weights[l].flatten()
            start = end
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l+1]
            params[start:end] = self.biases[l].flatten()
            start = end
        return params

class FNN(NeuralNet):
    def __init__(self, units_per_layer, activation="sigmoid", cost="MSE", d_cost=None):
        super(FNN, self).__init__(units_per_layer, activation, cost, d_cost)

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
            outputs.append(self.activation[l](np.matmul(outputs[-1], self.weights[l]) + self.biases[l]))
        return outputs


class RNN(NeuralNet):
    def __init__(self, units_per_layer, activation="sigmoid", cost="MSE", d_cost=None):
        super(RNN, self).__init__(units_per_layer, activation, cost, d_cost)

        # recurrent weights
        self.recurrent_weights = []
        for l in np.arange(1, self.num_layers - 2):
            d = -1 / np.sqrt(self.units_per_layer[l] + 1)
            self.recurrent_weights.append(
                np.random.uniform(
                    low=-d,
                    high=d,
                    size=[self.units_per_layer[l], self.units_per_layer[l]],
                )
            )

        self.states = [np.zeros(l) for l in units_per_layer[1:-1]]

    def forward(self, inputs):
        """ Forward propagate the given inputs through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network

        RETURNS:
        outputs: (list or matrix) output for all layers for all inputs
        """
        inputs = np.asarray(inputs)
        outputs = [inputs]

        # recurrent layers
        for l in np.arange(1, self.num_layers - 1):
            print(l)
            total_in = np.dot(outputs[l - 1], self.weights[l - 1])
            total_in += np.dot(self.states[l - 1], self.recurrent_weights[l - 1])
            outputs.append(self.activation[l - 1](total_in))
            self.states[l - 1] = outputs[-1]

        # final layer
        print(l)
        total_in = np.dot(outputs[-1], self.weights[-1])
        outputs.append(self.activation[-1](total_in))

        return outputs
