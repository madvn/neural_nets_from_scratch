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
        self.feedback_net = FNN(list(reversed(units_per_layer)), activation=activation)

    def single_update(self, input, desired_output):
        """ one update for one input using feedback network """
        output = self.forward(input)
        err = desired_output - output[-1]
        feedback = self.feedback_net.forward(err)

        outer_deriv = np.reshape(self.d_activation[-1](input), [self.units_per_layer[-1],1])
        delta_w = np.outer(outer_deriv, feedback[0])
        self.weights[-1] += self.learning_rate * delta_w
        for l in reversed(range(0,self.num_layers-2)):
            self.weights[l] += self.learning_rate * np.outer(output[l], feedback[-l-2])

        return err

    def training_step(self, inputs, desired_outputs):
        """ One complete forward and backward pass through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        """
        inputs = np.asarray(inputs)
        desired_outputs = np.asarray(desired_outputs)
        error = []

        if len(np.shape(inputs)) > 1:
            for sample, desired_output in zip(inputs, desired_outputs):
                error.append(self.single_update(sample, desired_output))
        else:
            error = [self.single_update(inputs, desired_outputs)]

        return error, np.sum(error)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        help="(string) directory to save results in",
        type=str,
        default="../results/backprop",
    )
    parser.add_argument(
        "--num_epochs",
        help="(int) number of training epochs",
        type=int,
        default=10000,
    )
    args = parser.parse_args()

    # data
    inputs = np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    outputs = np.asarray(
        [[-1, -1], [1, 1], [1, 1], [1, -1]]
    )  # 2 outputs - OR and XOR logic gates

    # craete network for feedback alignment training
    net = FeedbackAlignment([2, 3, 2, 2], activation="tanh")
    for e in range(args.num_epochs):
        _, e = net.training_step(inputs, outputs)
        print("Error: {}".format(e))

    # final fwd ppass
    print("Outputs\n", net.forward(inputs)[-1])
