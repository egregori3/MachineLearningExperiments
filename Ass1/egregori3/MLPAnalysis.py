"""
Please see README.txt for list of code sources
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

"""

from sklearn.neural_network import MLPClassifier
from PlotClassifiers import PlotClassifiers


PlotThese = [
# un-comment to run GridSearchCV parameters
#                {'dataset':'wifi', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
#                {'dataset':'wifi2', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
#                {'dataset':'wifi3', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
#                {'dataset':'letter', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves

# Validation WiFi
#                {'dataset':'wifi','default':'manual','vc_name':'hidden_layer_sizes','vc_range':[10,25,50],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
#                {'dataset':'wifi2','default':'manual','vc_name':'hidden_layer_sizes','vc_range':[10,25,50],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
#                {'dataset':'wifi3','default':'manual','vc_name':'hidden_layer_sizes','vc_range':[10,25,50],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

# Validation Letter
#               {'dataset':'letter','default':'manual',
#                'vc_name':'hidden_layer_sizes','vc_range':[x for x in range(5,100,5)],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

# Final
#                {'dataset':'wifi', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
#                {'dataset':'wifi2', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves

#                {'dataset':'letter', 'default':'manual',
#                'hidden_layer_sizes':90,
#                'type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
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
manual_params =  {
                            'hidden_layer_sizes':10,
                            'activation':'relu',
                            'solver':'adam',
                            'learning_rate':'constant',
                            'learning_rate_init':0.010,
                            'max_iter':200
                }

def CreateClassifier(dop):
    cls = MLPClassifier(   hidden_layer_sizes=dop['hidden_layer_sizes'],
                            activation=dop['activation'],
                            solver=dop['solver'],
                            learning_rate=dop['learning_rate'],
                            learning_rate_init=dop['learning_rate_init'],
                            max_iter=dop['max_iter']
                        )
    print(cls.get_params())
    return cls


def run():
    # -----------------------------------------------------------------------------
    # Call engine
    # -----------------------------------------------------------------------------
    PlotClassifiers(    PlotThese,
                        CreateClassifier,
                        MLPClassifier(),
                        kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)


if __name__ == '__main__':
    run()


