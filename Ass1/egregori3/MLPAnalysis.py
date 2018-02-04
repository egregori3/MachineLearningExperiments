"""
Please see README.txt for list of code sources
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

"""

from sklearn.neural_network import MLPClassifier
from PlotClassifiers import PlotClassifiers


PlotThese = [
                {'dataset':'wifi', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
            ]


# -----------------------------------------------------------------------------
# Parameters to Tune
# -----------------------------------------------------------------------------
kfolds = 3
test_size = 0.3
prefix = 'MLP'
scores = ['accuracy']
lop = ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate', 'learning_rate_init', 'max_iter']
tuned_parameters =  [
                         {
                            'hidden_layer_sizes':[10,25,50],
                            'activation':['relu'],
                            'solver':['sgd', 'adam'],
                            'learning_rate':['constant'],
                            'learning_rate_init':[0.010],
                            'max_iter':[150, 200, 250]
                        }
                    ]
manual_params =  [
                         {
                            'hidden_layer_sizes':50,
                            'activation':'relu',
                            'solver':'adam',
                            'learning_rate':'constant',
                            'learning_rate_init':0.010,
                            'max_iter':200,
                        }
                    ]

def CreateClassifier(dop):
    return MLPClassifier(   hidden_layer_sizes=dop['hidden_layer_sizes'],
                            activation=dop['activation'],
                            solver=dop['solver'],
                            learning_rate=dop['learning_rate'],
                            learning_rate_init=dop['learning_rate_init'],
                            max_iter=dop['max_iter']
                        )


# -----------------------------------------------------------------------------
# Call engine
# -----------------------------------------------------------------------------
PlotClassifiers(    PlotThese,
                    CreateClassifier,
                    MLPClassifier(),
                    kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)



