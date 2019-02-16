import numpy as np

from neural_nets.models import FNN


class FeedbackAlignment(FNN):
    """ Lillicrap, Timothy P., et al. Random synaptic feedback weights support
    error backpropagation for deep learning. Nature communications 7 (2016): 13276."""

    def __init__(
        self,
        units_per_layer,
        learning_rate=0.01,
        activation="sigmoid",
        cost="MSE",
        d_cost=None,
    ):
        """ Create FNN for training with backprop based on specifications

        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        learning_rate: (float) learning_rate to update the weights and biases of the networks
        activation: (str or list of strs ,len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid
        cost: (str or lambda) Choose the cost function to use, currently only 'MSE'. Alternatively provide lambda that takes two args, desired_outputs and actual_outputs in that order
        d_cost: (lambda) If cost is a lambda, provide another lambda that also takes same two args to estimate the derivative of cost function with respect to output

        Note: weights are uniform randomly initialized in [-1/sqrt(d), 1/sqrt(d)] where d is the number of inputs a neuron receives
        """
        super(FeedbackAlignment, self).__init__(
            units_per_layer, learning_rate, activation, cost, d_cost
        )

        # initializing feedback netowrk with random weights
        self.feedback_net = FNN(reversed(units_per_layer), activation=activation)

    def training_step(self, inputs, desired_outputs):
        """ One complete forward and backward pass through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        """
        inputs = np.asarray(inputs)
        desired_outputs = np.asarray(desired_outputs)

        outputs = self.forward(inputs)
        feedback = self.feedback_net.forward(desired_outputs - outputs)
