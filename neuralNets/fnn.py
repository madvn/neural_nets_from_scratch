import numpy as np
import matplotlib.pyplot as plt

class FNN:
    def __init__(self, units_per_layer, learning_rate=0.01, activation='sigmoid'):
        """ Create FNN based on specifications

        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        learning_rate: (float) learning_rate to update the weights and biases of the networks
        activation: (str or list of strs ,len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid

        Note: weights are uniform randomly initialized in [-1/sqrt(d), 1/sqrt(d)] where d is the number of inputs a neuron receives
        """
        assert isinstance(units_per_layer,list) and len(units_per_layer)>=2, "units_per_layer should be a list of len greater than 2 i.e. minimum of 1 input and 1 output layer"
        self.num_layers = len(units_per_layer)
        self.units_per_layer = units_per_layer
        self.learning_rate = learning_rate

        # lambdas for supported activation functions
        self.activation_funcs = {
                                'sigmoid': lambda x:1/(1+np.exp(-x)),
                                'tanh': lambda x:2/(1+np.exp(-2*x)) - 1,
                                'relu': lambda x:np.maximum(0,x),
                                }
        # lambdas for derivatives of supported activation functions
        self.d_activation_funcs = {
                                'sigmoid': lambda x:np.asarray(x)*(1-np.asarray(x)),
                                'tanh': lambda x:1-(np.asarray(x)**2),
                                'relu': lambda x:1 if x>0 else 0,
                                }

        ## setting activation for each layer
        # to verify if implementation exists for provided activation function
        def validate_activation_str(act):
            if act not in self.activation_funcs.keys():
                raise ValueError('Choose activation from {}'.format(self.activation_funcs.keys()))
        if isinstance(activation, str):
            # if activation is a string use same activation function for all layers
            activation = activation.strip()
            validate_activation_str(activation)
            self.activation = [self.activation_funcs[activation]]*(self.num_layers-1)
            self.d_activation = [self.d_activation_funcs[activation]]*(self.num_layers-1)
        else:
            # else activation must be a list of str correspondong to each layer
            assert len(activation)==self.num_layers-1, "if activation is a list, len(activation) should be equal to len(units_per_layer)-1 (first layer is input) OR activation should be str to assign same activation to all layers"
            self.activation = []
            # activation is a list
            for act in activation:
                validate_activation_str(act)
                self.activation.append(self.activation_funcs[act])
                self.d_activation.append(self.d_activation_funcs[act])

        # assigning intial random weights and biases based on layer size
        self.weights = []
        self.biases = []
        for l in np.arange(self.num_layers-1):
            d = -1/np.sqrt(self.units_per_layer[l]+1)
            self.weights.append(np.random.uniform(low = -d,
                                        high = d,
                                        size=[self.units_per_layer[l], self.units_per_layer[l+1]]
                                        ))
            self.biases.append(np.random.uniform(low = -d,
                                        high = d,
                                        size=[self.units_per_layer[l+1]]
                                        ))

    def forward(self, inputs):
        """ Forward propogate the given inputs through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network

        RETURNS:
        outputs: (list or matrix) one output correspondong to each input
        """
        inputs = np.asarray(inputs)
        outputs = [inputs]
        for l in np.arange(self.num_layers-1):
            if outputs[-1].ndim == 1: outputs[-1] = [outputs[-1]]
            outputs.append(self.activation[l](np.matmul(outputs[-1], self.weights[l]) + self.biases[l]))
        return outputs

    def backward(self, inputs, desired_outputs, outputs):
        """ Backward propagation of error through the network

        ARGS
        inputs: (list or matrix) one row for each input to the network
        desired_outputs: (list or matrix) one row for desired output for each input
        outputs:  (list or matrix) one row for actual output to each input
        """
        # average MSE along each dimension
        error = np.mean((desired_outputs - outputs[-1])**2, 0)
        del_l = -(desired_outputs - outputs[-1]) * self.d_activation[-1](outputs[-1])

        delta_l_w = []
        delta_l_b = []
        # compute gradients
        for l in reversed(np.arange(1,self.num_layers)):
            # chnage in w_l = del_l * outputs_l-1
            d_w = np.transpose(np.matmul(np.transpose(del_l), outputs[l-1]))
            # change_in_b_l = del_l * ones (since signal coming through bias = 1)
            d_b = np.sum(del_l, 0)

            # collecting these changes to apply them in the end
            delta_l_w.append(d_w)
            delta_l_b.append(d_b)

            # estimating del_l for next layer as follows del_l-1 = del_l * w_l-1 * f'(outputs[l-1])
            del_l = np.multiply(np.matmul(del_l, np.transpose(self.weights[l-1])), self.d_activation[l-1](outputs[l-1]))

        # apply gradients
        for l in reversed(range(self.num_layers-1)):
            self.weights[l] -= np.multiply(self.learning_rate, np.asarray(delta_l_w[-l-1]))
            self.biases[l] -= np.multiply(self.learning_rate, np.asarray(delta_l_b[-l-1].flatten()))

        return error, np.sum(error)


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
    nn = FNN([2,3,2],activation='tanh')
    inputs = [[-1,-1],[-1,1],[1,-1],[1,1]]
    outputs = [[-1,-1],[1,1],[1,1],[1,-1]]
    errs = []
    for _ in range(5000):
        s = np.random.randint(4) # to pick one random input to train on
        error = nn.training_step(inputs, outputs)
        errs.append(error[1])
        print("ERROR : ", error[1])

    print("\n\n###############################################")
    print("After training")
    print("Inputs:\n", inputs)
    print("Outputs:\n", nn.forward(inputs)[-1])

    plt.figure()
    plt.plot(errs)
    plt.title("Training curve")
    plt.xlabel("Training steps")
    plt.ylabel('Error')
    plt.show()
