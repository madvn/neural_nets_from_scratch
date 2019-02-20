import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt


class KohonenMap:
    def __init__(self, map_size, input_size, learn_rate=0.01):
        """ Initialize kohonen map with random weights

        ARGS
        ------
        map_size - list len=2 - grid-size of map
        input_size - int - dimensionality of input
        learn_rate - double - learning rate to update weights
        """
        print(os.path.dirname(os.path.abspath(__file__)))
        self.map_size = map_size
        self.input_size = input_size
        self.learning_rate = learn_rate

        self.input_weights = np.random.random(
            [self.map_size[0], self.map_size[1], self.input_size]
        )
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                self.input_weights[i][j][:] /= np.linalg.norm(
                    self.input_weights[i][j][:]
                )

        self.lateral_weights = np.random.random(self.map_size)
        self.lateral_excitatory_mask = [
            [i, j]
            for i, j in itertools.product(np.arange(-3, 4, 1), repeat=2)
            if np.linalg.norm([i, j]) < 3
        ]
        self.lateral_inhibitory_mask = [
            [i, j]
            for i, j in itertools.product(np.arange(-8, 9, 1), repeat=2)
            if (np.linalg.norm([i, j]) >= 3 and np.linalg.norm([i, j]) < 8)
        ]

        self.activity = np.random.random(self.map_size)

        self.total_input = None
        # self.fig = None

    def _estimate_external_input(self, input):
        """ returns dot(input, weights)
        which is = cos(angle between input and weights)
        if both weights and inputs are normalized
        a.b = |a||b|cos(theta) = 1*1*cos(theta)
        """
        """external_input = np.zeros(self.map_size)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                external_input[i][j] = np.dot(self.input_weights[i][j][:], input)
        """
        reshaped_weights = np.reshape(
            self.input_weights, [np.prod(self.map_size), self.input_size]
        )
        input = np.reshape(input, [self.input_size, 1])
        external_input = np.matmul(reshaped_weights, input)
        external_input = np.reshape(external_input, self.map_size)
        return external_input

    def _estimate_lateral_input(self, input, external_activation):
        """ lateral input from within the map
        with connections being wrapped around """
        lateral_input = np.zeros(self.map_size)
        for i in range(map_size[0]):
            for j in range(map_size[1]):
                # adding in the excitatory inputs
                for inds in self.lateral_excitatory_mask:
                    ind_x, ind_y = self._wrap_indices(i, j, inds)
                    lateral_input[i][j] += external_activation[ind_x][ind_y] * (8)

                # adding in the inhibitory inputs
                for inds in self.lateral_inhibitory_mask:
                    ind_x, ind_y = self._wrap_indices(i, j, inds)
                    lateral_input[i][j] += external_activation[ind_x][ind_y] * (-1)

        return lateral_input

    def _activation(self, total_input):
        """ piece-wise sigmoid activation function
        f(x) = {
                0, if x <= 0
                5, if x >= 5
                x. otherwise
                }
        """
        total_input[total_input < 0] = 0
        total_input[total_input > 5] = 5
        return total_input

    def _wrap_indices(self, i, j, inds):
        def _wrap(ind, limit):
            if ind >= limit:
                ind = ind - limit
            if ind < 0:
                ind = ind + limit
            return ind

        ind_x = _wrap(i + inds[0], self.map_size[0])
        ind_y = _wrap(j + inds[1], self.map_size[1])
        return ind_x, ind_y

    def stimulate(self, input):
        # normalizing the input
        # input = np.asarray(input)
        # input = input / np.linalg.norm(input)
        external_activation = self._estimate_external_input(input)
        lateral_input = self._estimate_lateral_input(input, external_activation)
        total_input = external_activation + lateral_input
        self.eta = self._activation(total_input)

    def train(self, input):
        # normalizing the input
        # input = np.asarray(input)
        # input = input / np.linalg.norm(input)
        self.stimulate(input)

        # update weights
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                self.input_weights[i][j][:] += (
                    self.learning_rate * self.eta[i][j] * np.double(input)
                )
                self.input_weights[i][j][:] /= np.linalg.norm(
                    self.input_weights[i][j][:]
                )

    def visualize(self, save_dir="../results/", suffix=""):
        """ Displays input weight matrix as an image
        Undefined behavior for input_size!=3
        """
        plt.figure()
        plt.imshow(self.input_weights)
        title = f"sofm_{suffix}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "{}.png".format(title)))
        plt.close()


if __name__ == "__main__":
    map_size = [50, 50]
    input_size = 3
    num_iterations = 501

    # create
    kmap = KohonenMap(map_size, input_size)

    # train
    for i in range(num_iterations):
        print(f"Iteration # {i} out of {num_iterations}", end="\r")
        input = np.zeros(input_size)
        # input[np.random.randint(low=0, high=input_size-1)] = 1
        input[np.random.choice([1, 2, 2])] = 1
        kmap.train(input)
        if i % 100 == 0:
            kmap.visualize(suffix=i)
