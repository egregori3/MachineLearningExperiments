"""
Please see README.txt for list of code sources
http://scikit-learn.org/0.16/modules/svm.html#
"""

from sklearn.svm import SVC
from PlotClassifiers import PlotClassifiers


PlotThese = [
# un-comment to run GridSearchCV parameters
                {'dataset':'wifi', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter', 'default':'best','type':['CM','LC=accuracy','LC=neg_mean_squared_error']}, # plot wifi CM and LC best parameters curves

# Validation
                {'dataset':'wifi','default':'manual','kernel':'rbf','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2','default':'manual','kernel':'rbf','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3','default':'manual','kernel':'rbf','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter','default':'manual','kernel':'rbf','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi','default':'manual','kernel':'poly','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2','default':'manual','kernel':'poly','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3','default':'manual','kernel':'poly','vc_name':'C','vc_range':range(1,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'letter','default':'manual',
                'kernel':'poly',
                'vc_name':'gamma','vc_range':range(0,20),'type':['VC=accuracy']}, # plot wifi CM and LC best parameters curves

#Final
                {'dataset':'wifi', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi2', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves
                {'dataset':'wifi3', 'default':'manual','type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves

                {'dataset':'letter', 'default':'manual',
                'kernel':'poly',
                'gamma':1,
                'type':['CM','LC=accuracy']}, # plot wifi CM and LC best parameters curves

            ]

# -----------------------------------------------------------------------------
# Parameters to Tune
# -----------------------------------------------------------------------------
kfolds = 3
test_size = 0.3
prefix = 'SVM'
scores = ['accuracy']
lop = ['C', 'kernel', 'degree', 'gamma']
tuned_parameters =  [
                         {
                            'C':[1,2,3],
                            'kernel':['rbf','poly'],
                            'degree':[1,2,3],
                            'gamma':[0,10,20],
                        }
                    ]
manual_params =  {
                            'C':1,
                            'kernel':'rbf',
                            'degree':2,
                            'gamma':10,
                }


def CreateClassifier(dop):
    return SVC(                 
                    C=dop['C'],
                    kernel=dop['kernel'],
                    degree=dop['degree'],
                    gamma=dop['gamma'],
                )


def run():
    # -----------------------------------------------------------------------------
    # Call engine
    # -----------------------------------------------------------------------------
    PlotClassifiers(    PlotThese,
                        CreateClassifier,
                        SVC(),
                        kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params)


if __name__ == '__main__':
    run()


