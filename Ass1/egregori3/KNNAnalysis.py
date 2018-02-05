"""
Please see README.txt for list of code sources
http://scikit-learn.org/0.16/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
"""

from sklearn.neighbors import KNeighborsClassifier
from PlotClassifiers import PlotClassifiers


PlotThese = [
# GridSearchCV
                {'dataset':'wifi', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves

# Validation wifi
                {'dataset':'wifi','default':'manual','vc_name':'n_neighbors','vc_range':range(2,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2','default':'manual','vc_name':'n_neighbors','vc_range':range(2,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3','default':'manual','vc_name':'n_neighbors','vc_range':range(2,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

# Validation Letter
                {'dataset':'letter','default':'manual',
                'vc_name':'n_neighbors','vc_range':range(2,50),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

#Final
                {'dataset':'wifi', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves

               {'dataset':'letter', 'default':'manual',
               'n_neighbors':4,
               'type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
            ]


# -----------------------------------------------------------------------------
# Parameters to Tune
# -----------------------------------------------------------------------------
kfolds = 3
test_size = 0.3
prefix = 'KNN'
scores = ['accuracy', 'neg_mean_squared_error']
lop = ['n_neighbors', 'weights', 'algorithm']
tuned_parameters =  [
                         {
                            'n_neighbors':[2,3,4,5,6,7,8,9,10],
                            'weights':['uniform','distance'],
                            'algorithm':['ball_tree','kd_tree','brute']
                        }
                    ]

manual_params =   {
                            'n_neighbors':7,
                            'weights':'uniform',
                            'algorithm':'ball_tree'
                    }


def CreateClassifier(dop):
    return KNeighborsClassifier(   n_neighbors=dop['n_neighbors'],
                                    weights=dop['weights'],
                                    algorithm=dop['algorithm']
                                )

# -----------------------------------------------------------------------------
# Call engine
# -----------------------------------------------------------------------------
PlotClassifiers(    PlotThese,
                    CreateClassifier,
                    KNeighborsClassifier(),
                    kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)



