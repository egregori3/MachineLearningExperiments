"""
Please see README.txt for list of code sources

http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

"""

from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from plot_learning_curve import plot_learning_curve
from sklearn.model_selection import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from LoadPreprocessDataset import LoadPreprocessDataset


# -----------------------------------------------------------------------------
# Load and preprocess dataset
# -----------------------------------------------------------------------------
X,y,name = LoadPreprocessDataset(sys.argv)

# -----------------------------------------------------------------------------
# Parameters to Tune
# -----------------------------------------------------------------------------
test_train_split = 0.3
kfolds = 3
scores = ['accuracy']
validation_param = 'max_depth'
validation_param_range = range(2,10)
tuned_parameters =  [
                         {
                            'C':[1,2,3,4,5,6,7,8,9,10],
                            'kernel':['rbf','poly'],
                            'degree':[1,2,3,4,5,6],
                            'gamma':[0,10,100],
                        }
                    ]

# -----------------------------------------------------------------------------
# Split dataset into training and test sets
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_train_split, random_state = 0)

# -----------------------------------------------------------------------------
# GridSearch() to find best parameters
# -----------------------------------------------------------------------------
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    bclf = GridSearchCV(SVC(), tuned_parameters, cv=kfolds,
                       scoring=score)
    bclf.fit(X_train, y_train)

    print("Best parameters set found on development set "+name)
    print()
    print(bclf.best_params_)
    print()
    if 0:
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in bclf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                % (mean_score, scores.std() * 2, params))
        print()

    print("Detailed classification report "+name)
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, bclf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

quit()

# -----------------------------------------------------------------------------
# Validation Curve
# -----------------------------------------------------------------------------
clf = DecisionTreeClassifier(   criterion=bclf.best_params_['criterion'],
                                splitter=bclf.best_params_['splitter'],
                                max_features=bclf.best_params_['max_features'],
                                max_depth=bclf.best_params_['max_depth'],
                                min_samples_split=bclf.best_params_['min_samples_split'],
                                min_samples_leaf=bclf.best_params_['min_samples_leaf']
                            )
train_scores, test_scores = validation_curve(
    clf, X, y, param_name=validation_param, param_range=validation_param_range,
    cv=kfolds, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("SVM Validation Curve"+name)
plt.xlabel("$\gamma$")
plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

# -----------------------------------------------------------------------------
# Learning Curve
# -----------------------------------------------------------------------------
plt = plot_learning_curve(clf, "SVM Learning Curve"+name, X,y, cv=kfolds)
plt.show()
