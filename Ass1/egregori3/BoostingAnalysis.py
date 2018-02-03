"""

"""


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from plot_learning_curve import plot_learning_curve
from sklearn.model_selection import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import sys
from LoadPreprocessDataset import LoadPreprocessDataset


X,y,name = LoadPreprocessDataset(sys.argv)


# -----------------------------------------------------------------------------
# Split dataset into training and test sets
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 0)

# -----------------------------------------------------------------------------
# Tune hyperparameters - Set the parameters by cross-validation
# -----------------------------------------------------------------------------
# criterion: It defines the function to measure the quality of a split. Sklearn supports “gini” criteria for Gini Index & “entropy” for Information Gain. By default, it takes “gini” value.
# splitter: It defines the strategy to choose the split at each node. Supports “best” value to choose the best split & “random” to choose the best random split. By default, it takes “best” value.
# max_features: It defines the no. of features to consider when looking for the best split. We can input integer, float, string & None value.
#     If an integer is inputted then it considers that value as max features at each split.
#     If float value is taken then it shows the percentage of features at each split.
#    If “auto” or “sqrt” is taken then max_features=sqrt(n_features).
#     If “log2” is taken then max_features= log2(n_features).
#     If None, then max_features=n_features. By default, it takes “None” value.
# max_depth: The max_depth parameter denotes maximum depth of the tree. It can take any integer value or None. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. By default, it takes “None” value.
# min_samples_split: This tells above the minimum no. of samples reqd. to split an internal node. If an integer value is taken then consider min_samples_split as the minimum no. If float, then it shows percentage. By default, it takes “2” value.
# min_samples_leaf: The minimum number of samples required to be at a leaf node. If an integer value is taken then consider min_samples_leaf as the minimum no. If float, then it shows percentage. By default, it takes “1” value.
# max_leaf_nodes: It defines the maximum number of possible leaf nodes. If None then it takes an unlimited number of leaf nodes. By default, it takes “None” value.
# min_impurity_split: It defines the threshold for early stopping tree growth. A node will split if its impurity is above the threshold otherwise it is a leaf.

kfolds = 3

tuned_parameters =  [
                         {
                            'n_estimators':[i for i in range(10,100,10)],
                        },

                    ]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    bclf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=kfolds,
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


# -----------------------------------------------------------------------------
# Validation Curve
# -----------------------------------------------------------------------------
param = 'n_estimators'
param_range = range(10,100,10)
clf = AdaBoostClassifier( n_estimators=bclf.best_params_['n_estimators'] )
train_scores, test_scores = validation_curve(
    clf, X, y, param_name=param, param_range=param_range,
    cv=kfolds, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("ADA Validation Curve"+name)
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
plt = plot_learning_curve(clf, "ADA Learning Curve"+name, X,y, cv=kfolds)
plt.show()




