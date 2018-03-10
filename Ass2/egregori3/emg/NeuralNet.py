# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import random
from math import exp

class NeuralNet:

    def __init__(self, l_rate, n_epoch, n_hidden):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.network = 0
        self.n_inputs = 0
        self.n_outputs = 0

#--------------------------------------------------------------------------------
# Forward propogate
#--------------------------------------------------------------------------------
    # Calculate neuron activation for an input
    def _activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation


    # Transfer neuron activation
    def _transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation)) 


    # Forward propagate input to a network output
    def _forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self._activate(neuron['weights'], inputs)
                neuron['output'] = self._transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs


#--------------------------------------------------------------------------------
# Backward propogate
#--------------------------------------------------------------------------------
    # Calculate the derivative of an neuron output
    def _transfer_derivative(self, output):
            return output * (1.0 - output) 


    # Backpropagate error and store in neurons
    def _backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self._transfer_derivative(neuron['output'])
        return (sum(errors)/len(errors))


    # Update network weights with error
    def _update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['delta']


#--------------------------------------------------------------------------------
# API
#--------------------------------------------------------------------------------
    # Output weights to console
    def dump_weights(self):
        print("Neural Net weights")
        for layer in self.network:
            print("layer:", end=" ")
            for neuron in layer:
                print(neuron['weights'])
        print()


    # Train a network for a fixed number of epochs
    def train_network_sgd(self, train):
        average_error_per_epoch = list()
        for epoch in range(self.n_epoch):
            for row in train:
                outputs = self._forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                expected[row[-1]] = 1
                average_error = self._backward_propagate_error(expected)
                self._update_weights(row)
            average_error_per_epoch.append(average_error)
        return average_error_per_epoch


    # Train a network for a fixed number of epochs
    def train_network_rhc(self, train):
        for epoch in range(self.n_epoch):
            for row in train:
                outputs = self._forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                expected[row[-1]] = 1
                self._backward_propagate_error(expected)
                self._update_weights(row)


    # Initialize a network
    def initialize_network(self, n_inputs, n_outputs):
#        print("Initialize Network: {},{},{}".format(n_inputs, self.n_hidden, n_outputs))
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(self.n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(self.n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.network = network


    # Make a prediction with a network
    def predict(self, row):
        outputs = self._forward_propagate(row)
        return outputs.index(max(outputs)) # one hot decoding


