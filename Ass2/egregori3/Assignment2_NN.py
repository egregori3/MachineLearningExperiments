from csv import reader
from emg.NeuralNet import NeuralNet
from emg.EvaluateClassifier import EvaluateClassifier
from emg.LoadPreprocessDataset import LoadPreprocessDataset
from emg.HillClimbingSimulatedAnnealing import test_hill_climbing



#test_hill_climbing()
#stop

#-----------------------------------------------------------------------
# Load dataset from csv file
#-----------------------------------------------------------------------
preprocess = LoadPreprocessDataset('.//DataSets//wifi.csv')
dataset = preprocess.GetDataset()


#-----------------------------------------------------------------------
# Neural Network hyperparameters
#-----------------------------------------------------------------------
l_rate = 0.01
n_epoch = 100
n_hidden = 10


#-----------------------------------------------------------------------
# Baseling SGD
#-----------------------------------------------------------------------
classifier = NeuralNet(l_rate, n_epoch, n_hidden)
initializer = classifier.initialize_network
trainer = classifier.train_network_sgd      # classifier training function
predicter = classifier.predict                # Classifier prediction function
Evaluator = EvaluateClassifier(initializer, dataset, trainer, predicter)

examples, training_scores, validation_scores = Evaluator.LearningCurve(10)
#print('Examples: %s' % examples)
#print('Training Scores: %s' % training_scores)
#print('Validation Scores: %s' % validation_scores)
print("SGD Learning Curve")
print("examples, training, validation")
for i in range(0,len(examples)):
    print("%s,%s,%s" % (examples[i],training_scores[i],validation_scores[i]))