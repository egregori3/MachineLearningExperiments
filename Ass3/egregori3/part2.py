# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://etav.github.io/python/scikit_pca.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
# https://piazza.com/class/jc2dhebfn0n2qo?cid=952
# http://scikit-learn.org/stable/modules/decomposition.html


from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def Plot2d(projected, chart_title, labels):
    plt.scatter(projected[:, 0], projected[:, 1],
                c=labels, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar();

    import uuid
    plt.suptitle(chart_title)
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------- PART 2 - Run PCA, ICA, RP, ?? on two datasets -----------------
#-------------- Apply the dimensionality reduction algorithms ----------------- 
#-------------- to the two datasets and describe what you see. --------------- 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part2( dataset ):
    print("Part 2 PCA - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))

#------------------------------------------------------------------------------
#  PCA - Visualize data
#------------------------------------------------------------------------------
    print("PCA Visualization")
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)
    Plot2d(projected, "PCA:"+dataset['name'], labels)

#------------------------------------------------------------------------------
#  PCA - Determeine best number of components
#------------------------------------------------------------------------------
    print("PCA Plot components versus variance")
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');

    import uuid
    plt.suptitle("PCA:"+dataset['name'])
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()

#------------------------------------------------------------------------------
#  ICA - Visualize data
#------------------------------------------------------------------------------
    print("ICA Visualization")
    ica = FastICA(n_components=2)
    projected = ica.fit_transform(X)
    Plot2d(projected, "ICA:"+dataset['name'], labels)

#------------------------------------------------------------------------------
#  Random Projection - Visualize data
#------------------------------------------------------------------------------
    print("Random Projection Visualization")
    transformer = random_projection.GaussianRandomProjection(2)
    projected = transformer.fit_transform(X)
    Plot2d(projected, "RP:"+dataset['name'], labels)

#------------------------------------------------------------------------------
#  Linear Discriminant Analysis- Visualize data
#------------------------------------------------------------------------------
    print("Linear Discriminant Analysis Visualization")
    transformer = LinearDiscriminantAnalysis(n_components=2)
    projected = transformer.fit_transform(X,labels)
    Plot2d(projected, "LDA:"+dataset['name'], labels)
