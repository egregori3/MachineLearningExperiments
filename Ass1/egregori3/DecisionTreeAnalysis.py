"""
Please see README.txt for list of code sources
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

from sklearn.tree import DecisionTreeClassifier
import sys
import matplotlib.pyplot as plt
from PlotLearningCurve import PlotLearningCurve
from LoadPreprocessDataset import LoadPreprocessDataset
from FindBestParameters import FindBestParameters
from DisplayValidationCurve import DisplayValidationCurve
from PlotConfusionMatrix import PlotConfusionMatrix


PlotThese = [
                {'type':'CM LC'}, # plot CM and LC best parameters curves
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
prefix = 'DT'
scores = ['accuracy']
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


def CreateClassifier(dop):
    return DecisionTreeClassifier(   criterion=dop['criterion'],
                                    splitter=dop['splitter'],
                                    max_features=dop['max_features'],
                                    max_depth=dop['max_depth'],
                                    min_samples_split=dop['min_samples_split'],
                                    min_samples_leaf=dop['min_samples_leaf']
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
best_params = FindBestParameters(   DecisionTreeClassifier(), 
                                    tuned_parameters, 
                                    kfolds, 
                                    scores, 
                                    name,
                                    X,y,test_size )


PlotClassifiers(PlotThese,plt)


