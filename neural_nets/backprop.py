import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from neural_nets.models import FNN


class BackPropNet(FNN):
    """ Back-propagation training for FNNs """

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
        super(BackPropNet, self).__init__(
            units_per_layer, learning_rate, activation, cost, d_cost
        )

    def backward(self, inputs, desired_outputs, outputs):
        """ Backward propagation of error through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        outputs:  (list or matrix) one row for actual output to each input

        RETURNS:
        error: Mean square error for given desired outputs and actual outputs
        """
        ## NOTES
        # w_l - weights from layer l-1 to l
        # b_l - biases to layer l
        # del_l - derivative of error with respect to net input to layer l = dE/dnet = dE/dout * dout/dnet
        # d_activation - derivative of activation function - see lines 30-64
        # delta_l_w - gradient of error with respect to weights = dE/dw
        # delta_l_b - gradient of error with respect to biases = dE/db

        # average MSE along each dimension
        error = self.cost(
            desired_outputs, outputs[-1]
        )  # np.mean((desired_outputs - outputs[-1])**2, 0)

        # Comptuing dE/dnet for outermost layer = dE/dout * dout/dnet
        # del_l = -(desired_outputs - outputs[-1]) * self.d_activation[-1](outputs[-1])
        del_l = self.d_cost(desired_outputs, outputs[-1]) * self.d_activation[-1](
            outputs[-1]
        )

        delta_l_w = []

        ## Compute gradients
        for l in reversed(np.arange(1, self.num_layers)):
            # change in weights = dE/dnet * dnet/dw = del_l * outputs_l-1
            # dE/dnet = del_l = dE/dout * dout/dnet
            # Therefore, change in w_l = del_l * dnet/dw = del_l * out_l-1
            d_w = np.transpose(np.matmul(np.transpose(del_l), outputs[l - 1]))

            # collecting these changes to apply them later
            delta_l_w.append(d_w)

            # Next, estimating dE/dnet for next (previous) layer as follows
            # del_l-1 = dE/dnet for l-1, but that depends on
            # a) weighted del_l from previous layer = del_l * w_l.T (transpose because back-propagating)
            # b) taking gradient over activation to get to net = f'(o_l-1) -> derivative of activation
            # This dE/dnet for l-1 = [del_l * w_l-1.T] .* f'(outputs[l-1]); * -> matrix mul, .* -> element-wise mul
            del_l = np.multiply(
                np.matmul(del_l, np.transpose(self.weights[l - 1])),
                self.d_activation[l - 1](outputs[l - 1]),
            )

        ## Apply gradients
        for l in reversed(range(self.num_layers - 1)):
            self.weights[l] -= np.multiply(
                self.learning_rate, np.asarray(delta_l_w[-l - 1])
            )

        return error, np.sum(np.dot(error, error))

    def training_step(self, inputs, desired_outputs):
        """ One complete forward and backward pass through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        """
        inputs = np.asarray(inputs)
        desired_outputs = np.asarray(desired_outputs)

        outputs = self.forward(inputs)
        error = self.backward(inputs, desired_outputs, outputs)
        return error


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
    args = parser.parse_args()

    # data
    inputs = np.asarray([[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
    outputs = np.asarray(
        [[-1, -1], [1, 1], [1, 1], [1, -1]]
    )  # 2 outputs - OR and XOR logic gates

    for train_samples_per_epoch in range(1, len(inputs) + 1):
        print("\n\n###############################################")
        print("Trainining with {} samples per epoch".format(train_samples_per_epoch))
        plt.figure()
        # 10 separate runs
        for _ in range(10):
            nn = BackPropNet([3, 3, 2], activation="tanh")
            errs = []
            for t in range(10000):
                train_inds = np.random.randint(
                    0, high=len(inputs), size=[train_samples_per_epoch]
                )
                error = nn.training_step(inputs[train_inds], outputs[train_inds])
                if t % 500 == 0:
                    print("Error at step {} = {}".format(t, error[1]))
                errs.append(error[1])

            # display results with one final fwd pass
            print("After training")
            print("Inputs:\n", inputs)
            print("Desired Outputs:\n", outputs)
            print("Outputs:\n", nn.forward(inputs)[-1])

            # Plot training curve
            plt.plot(errs, alpha=0.3)
            plt.title(
                "Training curve for 10 runs where \ngradient is computed on {} inputs at a time".format(
                    train_samples_per_epoch
                )
            )
            plt.xlabel("Training steps")
            plt.ylabel("Error")

        plt.tight_layout()

        # save
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        plt.savefig(
            os.path.join(
                args.save_dir,
                "and_xor_{}_sample_training.png".format(train_samples_per_epoch),
            )
        )
        plt.close()
