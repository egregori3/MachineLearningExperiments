# https://machinelearningmastery.com/implement-backpropagation-classifier-scratch-python/

from random import randrange


class EvaluateClassifier:

    def __init__(self, initializer, dataset, trainer, predicter):
        self.initializer = initializer
        self.dataset    = dataset
        self.trainer    = trainer
        self.predicter  = predicter


    # Split a dataset into k folds
    def _cross_validation_split(self, n_folds):
        dataset_split = list()
        dataset_copy = list(self.dataset)
        fold_size = int(len(self.dataset) / n_folds)
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


    # Test the classifier
    def _test_classifier(self, test):
        predictions = list()
        for row in test:
            prediction = self.predicter(row)
            predictions.append(prediction)
        return(predictions)


    def _init_classifier(self):
        n_inputs = len(self.dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in self.dataset])) # One hot encoding
        network = self.initializer(n_inputs, n_outputs)
        return network


    # Initialize classifier
    def _init_train_classifier(self, train):
        network = self._init_classifier()
        self.trainer(train)
#        cls.dump_weights(network)
        return network


    # k-fold cross validation
    def CrossValidate(self, n_folds):
        folds = self._cross_validation_split(n_folds)
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

            self._init_train_classifier(train_set)
            validation_prediction = self._test_classifier(test_set)
            training_prediction = self._test_classifier(train_set)
            validation_scores.append(self._accuracy_metric(test_actual, validation_prediction))
            training_scores.append(self._accuracy_metric(train_actual, training_prediction))

        return training_scores, validation_scores


    def LearningCurve(self, kfolds):
        # create/init network
        self._init_classifier()

        # break dataset up into folds
        folds = self._cross_validation_split(kfolds)

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

            self.trainer(train_set)
            validation_prediction = self._test_classifier(test_set)
            validation_scores.append(self._accuracy_metric(test_actual, validation_prediction))

            training_prediction = self._test_classifier(train_set)
            training_scores.append(self._accuracy_metric(train_actual, training_prediction))

        return examples, training_scores, validation_scores

