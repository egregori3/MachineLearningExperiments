"""
Please see README.txt for list of code sources
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

from sklearn.tree import DecisionTreeClassifier
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
prefix = 'DT'
scores = ['accuracy', 'neg_mean_squared_error']
lop = ['criterion', 'splitter', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf']
tuned_parameters =  [
                         {
                            'criterion':['entropy','gini'],
                            'splitter':['best','random'],
                            'max_features':['auto','sqrt','log2',None],
                            'max_depth':[3,4,5,6,None],
                            'min_samples_split':[2,3,4],
                            'min_samples_leaf':[1,2,3,4]
                        }
                    ]
manual_params =   {
                            'criterion':'entropy',
                            'splitter':'best',
                            'max_features':'auto',
                            'max_depth':None,
                            'min_samples_split':2,
                            'min_samples_leaf':1
                    }


def CreateClassifier(dop):
    return DecisionTreeClassifier(   criterion=dop['criterion'],
                                    splitter=dop['splitter'],
                                    max_features=dop['max_features'],
                                    max_depth=dop['max_depth'],
                                    min_samples_split=dop['min_samples_split'],
                                    min_samples_leaf=dop['min_samples_leaf']
                                )

# -----------------------------------------------------------------------------
# Call engine
# -----------------------------------------------------------------------------
PlotClassifiers(    PlotThese,
                    CreateClassifier,
                    DecisionTreeClassifier(),
                    kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)
