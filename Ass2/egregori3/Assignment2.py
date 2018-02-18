from csv import reader
from emg.NeuralNet import NeuralNet
from emg.EvaluateClassifier import EvaluateClassifier
from emg.LoadPreprocessDataset import LoadPreprocessDataset


# Load File
preprocess = LoadPreprocessDataset('.//DataSets//wifi.csv')
dataset = preprocess.GetDataset()


# evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 10
n_hidden = 10


Evaluator = EvaluateClassifier()
training_scores, validation_scores = Evaluator.CrossValidate( NeuralNet(), dataset, n_folds, l_rate, n_epoch, n_hidden)
print('Training Scores: %s' % training_scores)
print('Validation Scores: %s' % validation_scores)
mean_training = sum(training_scores)/float(len(training_scores))
mean_validation = sum(validation_scores)/float(len(validation_scores))
print('Mean Accuracy: %.3f%%   %.3f%%' % (mean_training,mean_validation) )