"""
Please see README.txt for list of code sources
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

"""

from sklearn.neural_network import MLPClassifier
import sys
import matplotlib.pyplot as plt
from PlotLearningCurve import PlotLearningCurve
from LoadPreprocessDataset import LoadPreprocessDataset
from FindBestParameters import FindBestParameters
from DisplayValidationCurve import DisplayValidationCurve
from PlotConfusionMatrix import PlotConfusionMatrix


PlotThese = [
                {'type':'CM LC SLC'}, # plot CM and LC best parameters curves
#                {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':3, 'min_samples_leaf':1},
 #               {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':3, 'min_samples_leaf':2},
  #              {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':3, 'min_samples_leaf':3},
   #             {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':3, 'min_samples_leaf':4},
    #            {'type':'VC', 'vc_name':'min_samples_leaf', 'vc_range':range(1,50), 'max_depth':3, 'min_samples_split':2},
     #           {'type':'VC', 'vc_name':'min_samples_leaf', 'vc_range':range(1,50), 'max_depth':3, 'min_samples_split':3},
      #          {'type':'VC', 'vc_name':'min_samples_leaf', 'vc_range':range(1,50), 'max_depth':3, 'min_samples_split':4},
       #         {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':4, 'min_samples_leaf':1},
        #        {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':4, 'min_samples_leaf':2},
         #       {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':4, 'min_samples_leaf':3},
          #      {'type':'VC', 'vc_name':'min_samples_split', 'vc_range':range(2,50), 'max_depth':4, 'min_samples_leaf':4},
           #     {'type':'VC', 'vc_name':'min_samples_leaf', 'vc_range':range(1,50), 'max_depth':4, 'min_samples_split':2},
            #    {'type':'VC', 'vc_name':'min_samples_leaf', 'vc_range':range(1,50), 'max_depth':4, 'min_samples_split':3},
             #   {'type':'VC', 'vc_name':'min_samples_leaf', 'vc_range':range(1,50), 'max_depth':4, 'min_samples_split':4},
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
                            'hidden_layer_sizes':[10,50,100],
                            'activation':['identity', 'logistic', 'tanh', 'relu'],
                            'solver':['lbfgs', 'sgd', 'adam'],
                            'learning_rate':['constant', 'invscaling', 'adaptive'],
                            'learning_rate_init':[0.001, 0.010],
                            'max_iter':[150, 200, 250]
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


def PlotClassifiers(list_of_dicts,plt):
    for params in list_of_dicts:
        top = best_params.copy()
        for parameter in lop:
            if parameter in params.keys():
                top[parameter] = params[parameter]
        clf = CreateClassifier( top )

        # -----------------------------------------------------------------------------
        # Put parameters in title
        # ----------------------------------------------------------------------------- 
        pvalues = ""
        add_lf = 50
        for pname in lop:
            pvalues += (pname+":"+str(top[pname])+",")
            if len(pvalues) > add_lf:
                pvalues += "\n"
                add_lf += 50

        # -----------------------------------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------------------------------- 
        if 'CM' in params['type']:
            title = name+" "+prefix+" Confusion Matrix"+"\n"+pvalues
            plt.figure()
            PlotConfusionMatrix(clf,X,y,test_size,classes,title=title)

        # -----------------------------------------------------------------------------
        # Learning Curve
        # -----------------------------------------------------------------------------
        if 'LC' in params['type']:
            title = name+" "+prefix+" Learning Curve"+"\n"+pvalues
            plt = PlotLearningCurve(clf, title, X, y, cv=kfolds)

        # -----------------------------------------------------------------------------
        # Learning Curve
        # -----------------------------------------------------------------------------
        if 'SLC' in params['type']:
            title = name+" "+prefix+" Learning Curve"+"\n"+pvalues
            plt = PlotLearningCurve(clf, title, sX, y, cv=kfolds)

        # -----------------------------------------------------------------------------
        # Validation Curve
        # -----------------------------------------------------------------------------
        if 'VC' in params['type']:
            title = name+" "+prefix+" Validation Curve"+"\n"+pvalues
            plt.figure()
            DisplayValidationCurve(clf, X, y, params['vc_name'], params['vc_range'], title, kfolds)

    plt.show()


# -----------------------------------------------------------------------------
# Load and preprocess dataset
# -----------------------------------------------------------------------------
X,y,name,sX,classes = LoadPreprocessDataset(sys.argv)


# -----------------------------------------------------------------------------
# find best parameters
# -----------------------------------------------------------------------------
best_params = FindBestParameters(   MLPClassifier(), 
                                    tuned_parameters, 
                                    kfolds, 
                                    scores, 
                                    name,
                                    X,y,test_size )

PlotClassifiers(PlotThese,plt)


