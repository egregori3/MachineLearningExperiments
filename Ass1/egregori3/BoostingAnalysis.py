"""
Please see README.txt for list of code sources
https://stackoverflow.com/questions/32210569/using-gridsearchcv-with-adaboost-and-decisiontreeclassifier
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys
import matplotlib.pyplot as plt
from PlotLearningCurve import PlotLearningCurve
from LoadPreprocessDataset import LoadPreprocessDataset
from FindBestParameters import FindBestParameters
from DisplayValidationCurve import DisplayValidationCurve
from PlotConfusionMatrix import PlotConfusionMatrix



PlotThese = [
                {'type':'CM LC'}, # plot CM and LC best parameters curves
            ]


# -----------------------------------------------------------------------------
# Parameters to Tune
# -----------------------------------------------------------------------------
kfolds = 3
test_size = 0.3
prefix = 'ADA'
scores = ['accuracy']
lop = ['n_estimators','criterion', 'splitter', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf']
tuned_parameters =  [
                         {
                            'n_estimators':[10, 50, 100],
                            'base_estimator__criterion':['entropy','gini'],
                            'base_estimator__splitter':['best','random'],
                            'base_estimator__max_features':['auto','sqrt','log2',None],
                            'base_estimator__max_depth':[3,4,5,6,None],
                            'base_estimator__min_samples_split':[2,3,4],
                            'base_estimator__min_samples_leaf':[1,2,3,4]
                        }
                    ]


def CreateClassifier(dop):
    DTC = DecisionTreeClassifier(   criterion=dop['criterion'],
                                    splitter=dop['splitter'],
                                    max_features=dop['max_features'],
                                    max_depth=dop['max_depth'],
                                    min_samples_split=dop['min_samples_split'],
                                    min_samples_leaf=dop['min_samples_leaf']
                                )

    return AdaBoostClassifier( base_estimator=DTC, n_estimators=dop['n_estimators'] )


def PlotClassifiers(list_of_dicts,plt):
    for params in list_of_dicts:
        top =   {
                    'n_estimators':best_params['n_estimators'],
                    'criterion':best_params['base_estimator__criterion'],
                    'splitter':best_params['base_estimator__splitter'],
                    'max_features':best_params['base_estimator__max_features'],
                    'max_depth':best_params['base_estimator__max_depth'],
                    'min_samples_split':best_params['base_estimator__min_samples_split'],
                    'min_samples_leaf':best_params['base_estimator__min_samples_leaf']
                }
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
best_params = FindBestParameters(   AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), 
                                    tuned_parameters, 
                                    kfolds, 
                                    scores, 
                                    name,
                                    X,y,test_size )

PlotClassifiers(PlotThese,plt)



