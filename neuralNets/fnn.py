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
    def __init__(self, units_per_layer, learning_rate=0.01, activation='sigmoid', cost='MSE',d_cost=None):
        """ Create FNN based on specifications

        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        learning_rate: (float) learning_rate to update the weights and biases of the networks
        activation: (str or list of strs ,len=len(units_per_layer)) Activation for each layer, choose from ['sigmoid', 'tanh', 'relu'], default = sigmoid
        cost: (str or lambda) Choose the cost function to use, currently only 'MSE'. Alternatively provide lambda that takes two args, desired_outputs and actual_outputs in that order
        d_cost: (lambda) If cost is a lambda, provide another lambda that also takes same two args to estimate the derivative of cost function with respect to output

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
                                'relu': lambda x:np.asarray([1 if i>0 else 0 for i in x]),
                                }
        # setting activation for each layer
        self._init_activations(activation)

        # lambdas for cost functions - lambda takes desired outputs d, and outputs o
        self.cost_funcs = {
                            'MSE': lambda d,o: np.mean(0.5*(np.asarray(d)-np.asarray(o))**2,0),
                            }

        # lambdas for derivatives of cost functions
        self.d_cost_funcs = {
                            'MSE': lambda d,o:-(d-o),
                            }
        # set cost function
        self._init_cost(cost, d_cost)

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

    def _init_activations(self, activation):
        """ Internal function to set activation function for each layer"""
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
            assert len(activation)==self.num_layers-1, ('if activation is a list,'
                    ' len(activation) should be equal to len(units_per_layer)-1, since first layer is input OR'
                    ' activation should be str to assign same activation to all layers')
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
            throw_away_lambda = lambda:0
            if ((isinstance(cost, type(throw_away_lambda)) and cost.__name__ == throw_away_lambda.__name__) and
                (isinstance(d_cost, type(throw_away_lambda)) and d_cost.__name__ == throw_away_lambda.__name__)):
                self.cost = cost
                self.d_cost = d_cost
            else:
                # no valid cost function has been defined
                raise ValueError('Choose cost function from {} or provide lambdas for cost and d_cost of __init__'.format(self.cost_funcs.keys()))
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
        error = self.cost(desired_outputs, outputs[-1]) #np.mean((desired_outputs - outputs[-1])**2, 0)

        # Comptuing dE/dnet for outermost layer = dE/dout * dout/dnet
        #del_l = -(desired_outputs - outputs[-1]) * self.d_activation[-1](outputs[-1])
        del_l = self.d_cost(desired_outputs,outputs[-1]) * self.d_activation[-1](outputs[-1])

        delta_l_w = []
        delta_l_b = []

        ## Compute gradients
        for l in reversed(np.arange(1,self.num_layers)):
            # change in weights = dE/dnet * dnet/dw = del_l * outputs_l-1
            # dE/dnet = del_l = dE/dout * dout/dnet
            # Therefore, change in w_l = del_l * dnet/dw = del_l * out_l-1
            d_w = np.transpose(np.matmul(np.transpose(del_l), outputs[l-1]))
            # Similarly, change in bias = del_l * ones (since signal coming through bias = 1)
            d_b = np.sum(del_l, 0)

            # collecting these changes to apply them later
            delta_l_w.append(d_w)
            delta_l_b.append(d_b)

            # Next, estimating dE/dnet for next (previous) layer as follows
            # del_l-1 = dE/dnet for l-1, but that depends on
            # a) weighted del_l from previous layer = del_l * w_l.T (transpose because back-propagating)
            # b) taking gradient over activation to get to net = f'(o_l-1) -> derivative of activation
            # This dE/dnet for l-1 = [del_l * w_l-1.T] .* f'(outputs[l-1]); * -> matrix mul, .* -> element-wise mul
            del_l = np.multiply(np.matmul(del_l, np.transpose(self.weights[l-1])), self.d_activation[l-1](outputs[l-1]))

        ## Apply gradients
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
    plt.figure()
    # 10 separate runs
    for _ in range(10):
        nn = FNN([2,3,2,2],activation='tanh')
        inputs = [[-1,-1],[-1,1],[1,-1],[1,1]]
        outputs = [[-1,-1],[1,1],[1,1],[1,-1]] # 2 outputs - OR and XOR logic gates
        errs = []
        for _ in range(10000):
            s1 = np.random.randint(4) # to pick one random input to train on
            s2 = np.random.randint(4) # to pick one random input to train on
            s3 = np.random.randint(4) # to pick one random input to train on
            #error = nn.training_step([inputs[s1],inputs[s2]], [outputs[s1],outputs[s2]])
            error = nn.training_step(inputs, outputs)
            errs.append(error[1])
            #print("ERROR : ", error[1])

        # display results with one final fwd pass
        print("\n\n###############################################")
        print("After training")
        print("Inputs:\n", inputs)
        print("Desired Outputs:\n", outputs)
        print("Outputs:\n", nn.forward(inputs)[-1])

        # Plot training curve
        plt.plot(errs, alpha=0.3)
        plt.title("Training curve for 10 runs where \ngradient is computed on all (4) inputs at a time")
        plt.xlabel("Training steps")
        plt.ylabel('Error')

    plt.tight_layout()
    #plt.savefig('./and_xor_4_sample_training.png')
    plt.show()
