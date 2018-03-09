# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import randrange


class EvaluateClassifier:

    # Split a dataset into k folds
    def _cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


    # Calculate accuracy percentage
    def _accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    # Test the algorithm
    def _test_algorithm( self, cls, network, test ):
        predictions = list()
        for row in test:
            prediction = cls.predict(network, row)
            predictions.append(prediction)
        return(predictions)


    # Initialize algorithm
    def _init_train_algorithm(self, cls, train, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        network = cls.initialize_network(n_inputs, n_hidden, n_outputs)
        cls.train_network(network, train, l_rate, n_epoch, n_outputs)
#        cls.dump_weights(network)
        return network

    # k-fold cross validation
    def CrossValidate(self, cls, dataset, n_folds, l_rate, n_epoch, n_hidden):
        folds = self._cross_validation_split(dataset, n_folds)
        validation_scores = list()
        training_scores = list()
        for fold in folds:
            train_set = list(folds)             # array of arrays
            train_set.remove(fold)              # remove one array
            train_set = sum(train_set, [])      # combine arrays

            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None

            test_actual = [row[-1] for row in fold]
            train_actual = [row[-1] for row in train_set]

            network = self._init_train_algorithm(cls, train_set, l_rate, n_epoch, n_hidden)
            validation_prediction = self._test_algorithm( cls, network, test_set )
            training_prediction = self._test_algorithm( cls, network, train_set )
            validation_scores.append(self._accuracy_metric(test_actual, validation_prediction))
            training_scores.append(self._accuracy_metric(train_actual, training_prediction))

        return training_scores, validation_scores


    def LearningCurve(self, cls, dataset, l_rate, n_epoch, n_hidden):
        # create/init network
        n_inputs = len(dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in dataset]))
        network = cls.initialize_network(n_inputs, n_hidden, n_outputs)

        # break dataset up into 5 folds
        kfolds = 10
        folds = self._cross_validation_split(dataset, kfolds)

        validation_scores = list()
        training_scores = list()
        examples = list()

        # test set is fold[-1]
        test_set = list()
        for row in folds[-1]:
            test_set.append(row)

        test_actual = [row[-1] for row in folds[-1]]

        # 0 - i folds = training, folds[-1] = test
        train_set = list()
        train_actual = list()
#        for i in [0]:
        for i in range(kfolds-1):
            for row in folds[i]:
                train_set.append(row)      # combine arrays
                train_actual.append(row[-1])

            examples.append(len(train_set))

            cls.train_network(network, train_set, l_rate, n_epoch, n_outputs)
            validation_prediction = self._test_algorithm( cls, network, test_set )
            validation_scores.append(self._accuracy_metric(test_actual, validation_prediction))

            training_prediction = self._test_algorithm( cls, network, train_set )
            training_scores.append(self._accuracy_metric(train_actual, training_prediction))

        return examples, training_scores, validation_scores

