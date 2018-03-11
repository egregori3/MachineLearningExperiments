from csv import reader
from emg.NeuralNet import NeuralNet
from emg.EvaluateClassifier import EvaluateClassifier
from emg.LoadPreprocessDataset import LoadPreprocessDataset
from emg.HillClimbingSimulatedAnnealing import test_hill_climbing


#-----------------------------------------------------------------------
# Neural Network hyperparameters
#-----------------------------------------------------------------------
l_rate = 0.01
n_hidden = 10


#-----------------------------------------------------------------------
# Create Neural Network based on hyperparameters
#-----------------------------------------------------------------------
def CreateNetwork(n_epoch, tf='sgd', parms={}):
    classifier = NeuralNet(l_rate, n_epoch, n_hidden)
    initializer = classifier.initialize_network
    if tf == 'sgd': trainer = classifier.train_network_sgd       # classifier training function
    if tf == 'rhc': trainer = classifier.train_network_rhc       # classifier training function
    if tf == 'sa': trainer = classifier.train_network_sa       # classifier training function
    predicter = classifier.predict                               # Classifier prediction function
    return EvaluateClassifier(initializer, dataset, trainer, predicter, parms)


#-----------------------------------------------------------------------
# Load dataset from csv file
#-----------------------------------------------------------------------
preprocess = LoadPreprocessDataset('.//DataSets//wifi.csv')
dataset = preprocess.GetDataset()


#-----------------------------------------------------------------------
# Baseling SGD - Training Error
#-----------------------------------------------------------------------
if 0:
    epochs = 500
    print("\nSGD Training Error %s epochs" % epochs)
    dump1, dump2, dump3, average_error_per_epoch = CreateNetwork(epochs).GetAccuracy(10,5)
    for i in range(len(average_error_per_epoch)):
        print("%s,%s" % (i,average_error_per_epoch[i]))


#-----------------------------------------------------------------------
# Baseling SGD - Finding accruacy
#-----------------------------------------------------------------------
if 0:
    print("\nSGD Accuracy")
    print("epochs, training, validation")
    for epochs in range(5,500,5):
        examples, training_scores, validation_scores, dump = CreateNetwork(epochs).GetAccuracy(10,5)
        print("%s,%s" % (epochs,validation_scores[0]))


#-----------------------------------------------------------------------
# Baseling SGD - Learning Curve
#-----------------------------------------------------------------------
if 0:
    epochs = 500
    Evaluator = CreateNetwork(epochs)
    examples, training_scores, validation_scores = Evaluator.LearningCurve(10)
    print("\nSGD Learning Curve %s epochs" % epochs)
    print("examples, training, validation")
    for i in range(0,len(examples)):
        print("%s,%s,%s" % (examples[i],training_scores[i],validation_scores[i]))


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# RHC - Training Error
#-----------------------------------------------------------------------
if 0:
    epochs = 100
    print("\nRHC Training Error %s epochs" % epochs)
    dump1, dump2, dump3, average_error_per_epoch = CreateNetwork(epochs,'rhc').GetAccuracy(10,1)
    for i in range(len(average_error_per_epoch)):
       print("%s,%s" % (i,average_error_per_epoch[i]))


#-----------------------------------------------------------------------
# RHC - Finding accruacy
#-----------------------------------------------------------------------
if 0:
    print("\nRHC Accuracy")
    print("epochs, training, validation")
    for epochs in range(50,500,50):
        examples, training_scores, validation_scores, dump = CreateNetwork(epochs,'rhc').GetAccuracy(10,1)
        print("%s,%s" % (epochs,validation_scores[0]))


#-----------------------------------------------------------------------
# RHC - Learning Curve
#-----------------------------------------------------------------------
if 0:
    epochs = 250
    Evaluator = CreateNetwork(epochs,'rhc')
    examples, training_scores, validation_scores = Evaluator.LearningCurve(10)
    print("\nRHC Learning Curve %s epochs" % epochs)
    print("examples, training, validation")
    for i in range(0,len(examples)):
        print("%s,%s,%s" % (examples[i],training_scores[i],validation_scores[i]))


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

parms = {'rate':0.9}

#-----------------------------------------------------------------------
# SA - Training Error
#-----------------------------------------------------------------------
if 0:
    epochs = 100
    print("\nSA Training Error %s epochs" % epochs)
    dump1, dump2, dump3, average_error_per_epoch = CreateNetwork(epochs,'sa', parms).GetAccuracy(10,1)
    for i in range(len(average_error_per_epoch)):
       print("%s" % (average_error_per_epoch[i]))


#-----------------------------------------------------------------------
# SA - Finding accruacy
#-----------------------------------------------------------------------
if 1:
    print("\nSA Accuracy")
    print("epochs, training, validation")
    for epochs in range(50,550,50):
        examples, training_scores, validation_scores, dump = CreateNetwork(epochs,'sa',parms).GetAccuracy(10,1)
        print("%s,%s" % (epochs,validation_scores[0]))


#-----------------------------------------------------------------------
# SA - Learning Curve
#-----------------------------------------------------------------------
if 0:
    epochs = 250
    Evaluator = CreateNetwork(epochs,'sa',parms)
    examples, training_scores, validation_scores = Evaluator.LearningCurve(10)
    print("\nSA Learning Curve %s epochs" % epochs)
    print("examples, training, validation")
    for i in range(0,len(examples)):
        print("%s,%s,%s" % (examples[i],training_scores[i],validation_scores[i]))
