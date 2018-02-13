# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from math import exp
from EvaluateClassifier import EvaluateClassifier


class NeuralNet:
    # Calculate neuron activation for an input
    def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train_network(network, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            for row in train:
                outputs = forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                backward_propagate_error(network, expected)
                update_weights(network, row, l_rate)

    # Initialize a network
    def initialize_network(n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    # Make a prediction with a network
    def predict(network, row):
        outputs = forward_propagate(network, row)
        return outputs.index(max(outputs))

    # Backpropagation Algorithm With Stochastic Gradient Descent
    def back_propagation(train, test, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        network = initialize_network(n_inputs, n_hidden, n_outputs)
        train_network(network, train, l_rate, n_epoch, n_outputs)
        predictions = list()
        for row in test:
            prediction = predict(network, row)
            predictions.append(prediction)
        return(predictions)

