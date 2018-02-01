"""
http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
http://scikit-learn.org/0.16/auto_examples/model_selection/grid_search_digits.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py


DecisionTreeClassifier(): This is the classifier function for DecisionTree. It is the main function for implementing the algorithms. Some important parameters are:

criterion: It defines the function to measure the quality of a split. Sklearn supports “gini” criteria for Gini Index & “entropy” for Information Gain. By default, it takes “gini” value.
splitter: It defines the strategy to choose the split at each node. Supports “best” value to choose the best split & “random” to choose the best random split. By default, it takes “best” value.
max_features: It defines the no. of features to consider when looking for the best split. We can input integer, float, string & None value.
    If an integer is inputted then it considers that value as max features at each split.
    If float value is taken then it shows the percentage of features at each split.
    If “auto” or “sqrt” is taken then max_features=sqrt(n_features).
    If “log2” is taken then max_features= log2(n_features).
    If None, then max_features=n_features. By default, it takes “None” value.
max_depth: The max_depth parameter denotes maximum depth of the tree. It can take any integer value or None. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. By default, it takes “None” value.
min_samples_split: This tells above the minimum no. of samples reqd. to split an internal node. If an integer value is taken then consider min_samples_split as the minimum no. If float, then it shows percentage. By default, it takes “2” value.
min_samples_leaf: The minimum number of samples required to be at a leaf node. If an integer value is taken then consider min_samples_leaf as the minimum no. If float, then it shows percentage. By default, it takes “1” value.
max_leaf_nodes: It defines the maximum number of possible leaf nodes. If None then it takes an unlimited number of leaf nodes. By default, it takes “None” value.
    min_impurity_split: It defines the threshold for early stopping tree growth. A node will split if its impurity is above the threshold otherwise it is a leaf.
"""


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from plot_learning_curve import plot_learning_curve
from sklearn.model_selection import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Set mydata to 'robot' or 'wifi'
# -----------------------------------------------------------------------------
mydata = 'robot' 

# -----------------------------------------------------------------------------
# Load dataset 
# -----------------------------------------------------------------------------
if not (mydata=='wifi' or mydata=='robot'):
    print("!ERROR! - Set dataset") 
    quite()

if mydata == 'wifi':
    print("Loading WIFI dataset")
    dataset = np.loadtxt("wifi_localization.txt", delimiter="\t")
    # -----------------------------------------------------------------------------
    # Preprocess dataset into (X)input and (y)output
    # -----------------------------------------------------------------------------
    X = dataset[:,0:7]
    y = dataset[:,7]


if mydata == 'robot':
    print("Loading ROBOT dataset")
    columns = []
    for i in range(1,25):
        columns.append('x'+str(i))
    columns.append('y1')
    df=pd.read_csv("sensor_readings_24.data", sep=',',header=None, names=columns)
    # -----------------------------------------------------------------------------
    # Preprocess dataset into (X)input and (y)output
    # convert category stings into numeric values
    # -----------------------------------------------------------------------------
    categories = {'Move-Forward':1, 'Sharp-Right-Turn':2, 'Slight-Right-Turn':3, 'Slight-Left-Turn':4}
    df['y1'] = df['y1'].apply(lambda y: categories[y])
    dataset = df.values
    X = dataset[:,0:24]
    y = dataset[:,24]

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
                            'criterion':['entropy'],
                            'splitter':['best'],
                            'max_features':['auto','sqrt','log2',None],
                            'max_depth':[3,4,5,6,None],
                            'min_samples_split':[2,3,4],
                            'min_samples_leaf':[1,2,3,4]
                        },
                        {
                            'criterion':['entropy'],
                            'splitter':['random'],
                            'max_features':['auto','sqrt','log2',None],
                            'max_depth':[3,4,5,6,None],
                            'min_samples_split':[2,3,4],
                            'min_samples_leaf':[1,2,3,4]
                        },
                        {
                            'criterion':['gini'],
                            'splitter':['best'],
                            'max_features':['auto','sqrt','log2',None],
                            'max_depth':[3,4,5,6,None],
                            'min_samples_split':[2,3,4],
                            'min_samples_leaf':[1,2,3,4]
                        },
                        {
                            'criterion':['gini'],
                            'splitter':['random'],
                            'max_features':['auto','sqrt','log2',None],
                            'max_depth':[3,4,5,6,None],
                            'min_samples_split':[2,3,4],
                            'min_samples_leaf':[1,2,3,4]
                        }
                    ]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    bclf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=kfolds,
                       scoring=score)
    bclf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
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

    print("Detailed classification report:")
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
param = 'max_depth'
param_range = range(2,10)
clf = DecisionTreeClassifier(   criterion=bclf.best_params_['criterion'],
                                splitter=bclf.best_params_['splitter'],
                                max_features=bclf.best_params_['max_features'],
                                max_depth=bclf.best_params_['max_depth'],
                                min_samples_split=bclf.best_params_['min_samples_split'],
                                min_samples_leaf=bclf.best_params_['min_samples_leaf']
                            )
train_scores, test_scores = validation_curve(
    clf, X, y, param_name=param, param_range=param_range,
    cv=kfolds, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve")
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
plt = plot_learning_curve(clf, "Learning Curve", X,y, cv=kfolds)
plt.show()

if 0:
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)

    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)

    y_pred = clf_gini.predict(X_test)
    y_pred_en = clf_entropy.predict(X_test)

    print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
    print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)



