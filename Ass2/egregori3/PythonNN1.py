# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch02/ch02.ipynb

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
import pandas as pd


def PlotData(X,y):
    #------------------------------------------------------------------------------
    # Plot input data (X,y)
    #------------------------------------------------------------------------------
    plt.figure()

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('./images/02_06.png', dpi=300)
    #plt.show()


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01):
        self.eta = eta
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []


    def fit(self, X, y, n_iter=1):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        for _ in range(n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.1):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    x_test = np.array([xx1.ravel(), xx2.ravel()]).T
#    print(x_test)
    Z = classifier.predict(x_test)
    Z = Z.reshape(xx1.shape)
    if 0:
        print()
        for row in Z:
            print()
            for col in row:
                if col>0: 
                    print('1', end='')
                else:
                    print('2', end='')
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)


if 0:
    #------------------------------------------------------------------------------
    # Read data into X,y
    #------------------------------------------------------------------------------
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                    'machine-learning-databases/iris/iris.data', header=None)
    print(df.tail())

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [1, 2]].values

X,y = make_moons(noise=0.3, random_state=0)

#------------------------------------------------------------------------------
# Fit (X,y) Neural Network
#------------------------------------------------------------------------------
ppn = Perceptron(eta=0.005)
for _ in range(0,200):
    ppn.fit(X, y)
#    print(ppn.predict(X))
    plot_decision_regions(X, y, classifier=ppn)

#------------------------------------------------------------------------------
# Plot error with respect to epoch
#------------------------------------------------------------------------------
plt.figure()
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)


plt.show()


