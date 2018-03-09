from csv import reader
from emg.NeuralNet import NeuralNet
from emg.EvaluateClassifier import EvaluateClassifier
from emg.LoadPreprocessDataset import LoadPreprocessDataset
from emg.HillClimbingSimulatedAnnealing import test_hill_climbing



#test_hill_climbing()
#stop


# Load File
preprocess = LoadPreprocessDataset('.//DataSets//wifi.csv')
dataset = preprocess.GetDataset()


# evaluate algorithm
l_rate = 0.01
n_epoch = 100
n_hidden = 10


Evaluator = EvaluateClassifier()
examples, training_scores, validation_scores = Evaluator.LearningCurve( NeuralNet(), dataset, l_rate, n_epoch, n_hidden)
print('Examples: %s' % examples)
print('Training Scores: %s' % training_scores)
print('Validation Scores: %s' % validation_scores)

print("examples, training, validation")
for i in range(0,len(examples)):
    print("%s,%s,%s" % (examples[i],training_scores[i],validation_scores[i]))