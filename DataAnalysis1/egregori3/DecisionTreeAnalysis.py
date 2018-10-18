"""
Please see README.txt for list of code sources
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

from sklearn.tree import DecisionTreeClassifier
from PlotClassifiers import PlotClassifiers


PlotThese = [
# Gridsearch
                {'dataset':'wifi', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves

# WiFi Validation
                {'dataset':'wifi','default':'manual','vc_name':'max_depth','vc_range':[3,4,5,6],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2','default':'manual','vc_name':'max_depth','vc_range':[3,4,5,6],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3','default':'manual','vc_name':'max_depth','vc_range':[3,4,5,6],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

# Letter Validation
                {'dataset':'letter','default':'manual',
                'max_depth':50,
                'min_samples_leaf':1,
                'vc_name':'min_samples_split','vc_range':[3,4,5,6,7,8,9,10],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

                {'dataset':'letter','default':'manual',
                'max_depth':50,
                'min_samples_leaf':5,
                'vc_name':'min_samples_split','vc_range':[3,4,5,6,7,8,9,10],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

                {'dataset':'letter','default':'manual',
                'max_depth':50,
                'min_samples_leaf':10,
                'vc_name':'min_samples_split','vc_range':[3,4,5,6,7,8,9,10],'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

# Final
                {'dataset':'wifi', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves

                {'dataset':'letter', 'default':'manual',
                'max_depth':50,
                'min_samples_leaf':1,
                'min_samples_split':4,
                'type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
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
                            'max_depth':[5,10,20,50,None],
                            'min_samples_split':[2,3,4],
                            'min_samples_leaf':[1,2,3,4]
                        }
                    ]
manual_params =   {
                            'criterion':'entropy',
                            'splitter':'best',
                            'max_features':'auto',
                            'max_depth':5,
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


def run():
    # -----------------------------------------------------------------------------
    # Call engine
    # -----------------------------------------------------------------------------
    PlotClassifiers(    PlotThese,
                        CreateClassifier,
                        DecisionTreeClassifier(),
                        kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)


if __name__ == '__main__':
    run()

