"""
Please see README.txt for list of code sources
https://stackoverflow.com/questions/32210569/using-gridsearchcv-with-adaboost-and-decisiontreeclassifier
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from PlotClassifiers import PlotClassifiers


PlotThese = [
                {'dataset':'wifi', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter', 'default':'manual','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
            ]


# -----------------------------------------------------------------------------
# Parameters to Tune
# -----------------------------------------------------------------------------
kfolds = 3
test_size = 0.3
prefix = 'ADA'
scores = ['accuracy']
lop = ['n_estimators','base_estimator__criterion', 'base_estimator__splitter', 'base_estimator__max_features', 'base_estimator__max_depth', 'base_estimator__min_samples_split', 'base_estimator__min_samples_leaf']
tuned_parameters =  [
                         {
                            'n_estimators':[10, 20, 30],
                            'base_estimator__criterion':['entropy','gini'],
                            'base_estimator__splitter':['best'],
                            'base_estimator__max_features':['sqrt','log2'],
                            'base_estimator__max_depth':[3],
                            'base_estimator__min_samples_split':[2],
                            'base_estimator__min_samples_leaf':[1]
                        }
                    ]
manual_params =  {
                            'n_estimators':100,
                            'base_estimator__criterion':'entropy',
                            'base_estimator__splitter':'best',
                            'base_estimator__max_features':'sqrt',
                            'base_estimator__max_depth':None,
                            'base_estimator__min_samples_split':2,
                            'base_estimator__min_samples_leaf':1,
                }
 


def CreateClassifier(dop):
    DTC = DecisionTreeClassifier(   criterion=dop['base_estimator__criterion'],
                                    splitter=dop['base_estimator__splitter'],
                                    max_features=dop['base_estimator__max_features'],
                                    max_depth=dop['base_estimator__max_depth'],
                                    min_samples_split=dop['base_estimator__min_samples_split'],
                                    min_samples_leaf=dop['base_estimator__min_samples_leaf']
                                )

    return AdaBoostClassifier( base_estimator=DTC, n_estimators=dop['n_estimators'] )


#-----------------------------------------------------------------------------
# Call engine
# -----------------------------------------------------------------------------
PlotClassifiers(    PlotThese,
                    CreateClassifier,
                    AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                    kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)
