# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ass1.PlotLearningCurve import PlotLearningCurve
from ass1.FindBestParameters import FindBestParameters
from ass1.DisplayValidationCurve import DisplayValidationCurve
from ass1.PlotConfusionMatrix import PlotConfusionMatrix


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
                            'solver':'sgd',
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


def PlotClassifiers(X, y, parms, name, classes):
    # -----------------------------------------------------------------------------
    # Split dataset into training and test sets
    # -----------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=0)

    if parms == 'best':
        best_params = FindBestParameters(   MLPClassifier(), 
                                            tuned_parameters, 
                                            kfolds, 
                                            scores, 
                                            name,
                                            X_train,y_train,
                                            X_test,y_test
                                        )
        top = best_params.copy()

    if parms == 'manual':
        top = manual_params.copy()

    clf = CreateClassifier( top )

    # -----------------------------------------------------------------------------
    # Generate classification report
    # ----------------------------------------------------------------------------- 
    print("Classification report")
    print("Parameters:")
    print(top)
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

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
    # Generate plots
    # ----------------------------------------------------------------------------- 
    # -----------------------------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------------------------- 
    title = name+" Confusion Matrix"+"\n"+pvalues
    PlotConfusionMatrix(clf,X_train,y_train,X_test,y_test,classes,title=title)

    # -----------------------------------------------------------------------------
    # Learning Curve
    # -----------------------------------------------------------------------------
    title = name+" Learning Curve "+'accuracy'+"\n"+pvalues
    plt = PlotLearningCurve(clf, title, X_train,y_train, cv=kfolds, scorer='accuracy')

#    # -----------------------------------------------------------------------------
#    # Validation Curve
#    # -----------------------------------------------------------------------------
#    title = name+" Validation Curve"+"\n"+pvalues
#    DisplayValidationCurve(clf, X_train,y_train, params['vc_name'], params['vc_range'], title, kfolds)

