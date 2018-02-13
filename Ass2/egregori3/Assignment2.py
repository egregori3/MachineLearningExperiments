from emg.LoadPreprocessDataset import LoadPreprocessDataset
from emg.NeuralNet import NeuralNet
from EvaluateClassifier import EvaluateClassifier


X_train, X_test, y_train, y_test, name, classes = LoadPreprocessDataset(['','wifi3'])
print("Classes: {}".format(classes))

# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))