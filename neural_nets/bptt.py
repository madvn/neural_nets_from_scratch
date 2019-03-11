import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from neural_nets.models import RNN


class BackPropThroTime(RNN):
    def __init__(
        self,
        units_per_layer,
        activation="sigmoid",
        cost="MSE",
        d_cost=None,
        bptt_window_size=50,
        learning_rate=0.01,
    ):
        super(BackPropThroTime, self).__init__(units_per_layer, activation)

    def backward(self, inputs, desired_outputs):
        outputs = self.forward(inputs)

    def training_step(self, inputs, desired_outputs):
        """ One complete forward and backward pass through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        """
        inputs = np.asarray(inputs)
        desired_outputs = np.asarray(desired_outputs)

        error = self.backward(inputs, desired_outputs)
        return error
