import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from neural_nets.models import FNN


class FeedbackAlignment(FNN):
    """ Lillicrap, Timothy P., et al. Random synaptic feedback weights support
    error backpropagation for deep learning. Nature communications 7 (2016): 13276."""

    def __init__(self, units_per_layer, activation="sigmoid", learning_rate=0.01):
        """ Create FNN for training with backprop based on specifications

        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        learning_rate: (float) learning_rate to update the weights and biases of the networks
        activation: (str or list of strs ,len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid

        Note: weights are uniform randomly initialized in [-1/sqrt(d), 1/sqrt(d)] where d is the number of inputs a neuron receives
        """
        super().__init__(units_per_layer, activation)
        self.learning_rate = learning_rate

        # initializing feedback netowrk with random weights
        self.feedback_net = FNN(list(reversed(units_per_layer)), activation="linear")

    def training_step(self, inputs, desired_outputs):
        """ One complete forward and backward pass through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        """
        desired_output = np.asarray(desired_outputs)

        # forward pass through net
        outputs = self.forward(inputs)

        # error
        err = desired_outputs - np.asarray(outputs[-1])

        # forward pass through feedback network
        feedback = self.feedback_net.forward(err)

        # back-propagating error - see backprop.py in the same dir for more details.
        # logic here is the same except we use feedback instead of deltas from previous layers
        for l in reversed(np.arange(1, self.num_layers)):
            # gradient through feedback
            del_l = np.multiply(
                feedback[self.num_layers - l - 1], self.d_activation[l - 1](outputs[l])
            )
            # gradient weighted by outputs of previous layer
            d_w = np.transpose(np.matmul(np.transpose(del_l), outputs[l - 1]))
            # weights updates
            self.weights[l - 1] += self.learning_rate * d_w

        return err, np.sum(np.asarray(err) ** 2)


if __name__ == "__main__":
    np.random.seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        help="(string) directory to save results in",
        type=str,
        default="../results/feedback_align",
    )
    parser.add_argument(
        "--num_epochs", help="(int) number of training epochs", type=int, default=100000
    )
    args = parser.parse_args()

    # data
    inputs = np.asarray([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    # 2 outputs - OR and XOR logic gates
    outputs = np.asarray([[0, 0], [1, 1], [1, 1], [1, 0]])

    plt.figure()

    for r in range(5):
        print("\nRun # {}".format(r))
        es = []
        # create network for feedback alignment training
        net = FeedbackAlignment([np.shape(inputs)[1], 10, np.shape(outputs)[1]], activation="sigmoid")

        for e in range(args.num_epochs):
            _, err = net.training_step(inputs, outputs)
            if e%5000 == 0:
                print("Error in epoch {}: {}".format(e, err))
            es.append(err)

        # final fwd ppass
        print("Outputs\n", net.forward(inputs)[-1])

        # plot
        plt.plot(es)

    # save fig
    plt.xlabel("Epoch #")
    plt.ylabel("SSE")
    plt.title("Feedback alignment performance in 5 runs")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    plt.savefig(os.path.join(args.save_dir, "fba_performance.png"))
