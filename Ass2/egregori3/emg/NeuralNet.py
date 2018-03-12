# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

import json
from random import random
from random import randint
from random import randrange
from math import exp
from copy import deepcopy

class NeuralNet:

    def __init__(self, l_rate, n_epoch, n_hidden):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.network = 0
        self.n_inputs = 0
        self.n_outputs = 0

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Forward propogate
#--------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------
# Backward propogate Stochastic Gradient Descent
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

    # Calculate the derivative of an neuron output
    def _transfer_derivative(self, output):
            return output * (1.0 - output) 


    def _sgd(self,layer,errors):
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * self._transfer_derivative(neuron['output'])


    # Backpropagate error and store in neurons
    def _backward_propagate_error_sgd(self, expected):
        average_output_error = 0
        average_output_error_divider = 0
        for i in reversed(range(len(self.network))):                        # propogate backwards
            layer = self.network[i]
            errors = list()

            if i != len(self.network)-1:
                for j in range(len(layer)):                                 # calculate hidden layer error
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):                                 # calculate output layer error
                    neuron = layer[j]
                    output_error = expected[j] - neuron['output']
                    errors.append(output_error)
                    average_output_error_divider += 1
                    average_output_error += output_error

            self._sgd(layer,errors)                                              # Adjust deltas based on errors

        return (average_output_error/average_output_error_divider)


    # Update network weights with error
    def _update_weights_sgd(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['delta']


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Training Utilities
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

    # Calculate black box error (treat NN as black box)
    def _black_box_error(self, row):
        outputs = self._forward_propagate(row)
        measured = outputs.index(max(outputs))
        actual = row[-1]
        return measured-actual


    # for each knob, calc current error, then adjust knob based on error
    def _turn_knobs(self, row, neuron, amount):
        for knob in range(len(neuron['weights'])):
            original = neuron['weights'][knob]
            initial_error = self._black_box_error(row)
            if initial_error < 0: neuron['weights'][knob] = original + amount
            if initial_error > 0: neuron['weights'][knob] = original - amount


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# API
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Visualization
#--------------------------------------------------------------------------------
    # Output weights to console
    def dump_weights(self):
        print("Neural Net weights")
        for layer in self.network:
            print("layer:", end=" ")
            for neuron in layer:
                print(neuron['weights'])
        print()


#--------------------------------------------------------------------------------
# Training
#--------------------------------------------------------------------------------

# SGD - Stochastic Gradient Descent

    def train_network_sgd(self, train, parms):
        average_error_per_epoch = list()
        for epoch in range(self.n_epoch):
            average_error = 0
            for row in train:
                outputs = self._forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                expected[row[-1]] = 1                           # one hot encoding
                average_error += self._backward_propagate_error_sgd(expected)
                self._update_weights_sgd(row)
            average_error_per_epoch.append(average_error/len(train))
        return average_error_per_epoch


# RHC - Randomized Hill Climbing

    def train_network_rhc(self, train, parms):
        average_error_per_epoch = list()
        for epoch in range(self.n_epoch):
            average_error = 0
            for row in train:
                for layer in self.network:
                    for neuron in layer:
                        self._turn_knobs(row, neuron, self.l_rate)
                average_error += self._black_box_error(row)
            average_error_per_epoch.append(average_error/len(train))
        return average_error_per_epoch


# SA - Simulated Annealing

    # RHC with decaying randomization
    def train_network_sa(self, train, params):
        average_error_per_epoch = list()
        min_error = 1000
        temperature = 1.0
        best_network = deepcopy(self.network)
        for epoch in range(self.n_epoch):
            average_error = 0
            bias = 10.0*random()*temperature
            for row in train:
                for layer in self.network:
                    for neuron in layer:
                        self._turn_knobs(row, neuron, self.l_rate*bias)
                average_error += self._black_box_error(row)
            average_error_per_epoch.append(average_error/len(train))
            temperature *= params['rate']
            if abs(average_error/len(train)) < min_error:
                 min_error = abs(average_error/len(train))   # improved, keep network
                 best_network = deepcopy(self.network)

        self.network = best_network
        return average_error_per_epoch


# GA - Genetic Algorithm

    def _create_hypothesis(self):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
        network.append(output_layer)
        return network


    def _create_population(self, size):
        population = []
        for i in range(size):
            population.append(self._create_hypothesis())
        return population


    def _evaluate_population(self, population, train):
        pop_errors = list()
        for hypothesis in population:
            self.network = hypothesis
            average_error = 0
            for row in train:
                average_error += self._black_box_error(row)
            pop_errors.append(average_error/len(train))
        return pop_errors


    def _crossover(self, population, number):
        # neurons are all in the same order.
        # must use deepcopy to avoid undesired dependencies
#        for i in range(int(round(number))):
            donor1 = randint(0,len(population)-1)
            donor2 = (donor1 + 1) % (len(population)-1)
            parent1 = population[donor1]
            parent2 = population[donor2]

            new_hypothesis = list()
            for neuron in range(len(parent1)):
                if random() > 0.5:
                    new_hypothesis.append(deepcopy(parent1[neuron]))
                else:
                    new_hypothesis.append(deepcopy(parent2[neuron]))

            population.append(new_hypothesis)


    # randomly replace weights with new random weights
    def _mutate(self, population, tomutate):
        for i in range(int(round(tomutate*len(population)))):
            network = population[randrange(0,len(population))]
            layer = network[randrange(0,len(network))]
            neuron = layer[randrange(0,len(layer))]
            weightid = randrange(0,len(neuron['weights']))
            neuron['weights'][weightid] = random() 


    # Kill worst performers
    def _prune(self, population, pop_error, replace):
        next_generation = list()
        worst = max([abs(x) for x in pop_error])
        mydelete = True
        for i in range(len(pop_error)):
            if mydelete and abs(pop_error[i]) == worst:
                mydelete = False
            else:
                next_generation.append(population[i]) # keep it

        return next_generation
 

    # Distributed hypothesis with mutation
    def train_network_ga(self, train, params):
        average_error_per_epoch = list()
        population = self._create_population(params['pop'])
        for epoch in range(self.n_epoch):
            average_error = 0
            average_count = 0

            # Evaluate
            pop_error = self._evaluate_population(population, train)

            # Kill poor performers
            population = self._prune(population, pop_error, params['replace'])

            # Create via mating
            self._crossover(population, params['mate']*len(population))

            # Mutate
            self._mutate(population, params['mutate'])

            average_error_per_epoch.append(sum(pop_error)/len(pop_error))

        return average_error_per_epoch


#--------------------------------------------------------------------------------
# Initialize network
#--------------------------------------------------------------------------------
    # Initialize a network
    def initialize_network(self, n_inputs, n_outputs):
#        print("Initialize Network: {},{},{}".format(n_inputs, self.n_hidden, n_outputs))
        if 0:
            network = list()
            hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(self.n_hidden)]
            network.append(hidden_layer)
            output_layer = [{'weights':[random() for i in range(self.n_hidden + 1)]} for i in range(n_outputs)]
            network.append(output_layer)
            with open('data.json', 'w') as fp:
                json.dump(network, fp)
        else:
            with open('data.json', 'r') as fp:
                network = json.load(fp)
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.network = network


#--------------------------------------------------------------------------------
# Predict with network
#--------------------------------------------------------------------------------
    # Make a prediction with a network
    def predict(self, row):
        outputs = self._forward_propagate(row)
        return outputs.index(max(outputs)) # one hot decoding


