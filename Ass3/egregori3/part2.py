# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://etav.github.io/python/scikit_pca.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
# https://piazza.com/class/jc2dhebfn0n2qo?cid=952
# http://scikit-learn.org/stable/modules/decomposition.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.kurtosis.html


from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def PlotDataFrame(df, title, subtitle, xlabel, ylabel):
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')
    # multiple line plot
    num=0
    for column in df.drop('x', axis=1):
        num+=1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

    # Add legend
    plt.legend(loc=2, ncol=2)

    # Add titles
    plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    import uuid
    plt.suptitle(subtitle)
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
    print("PART 2 - "+dataset['name'])
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

    plt.bar([i for i in range(1,len(pca.explained_variance_)+1)], pca.explained_variance_)
    plt.xlabel('component')
    plt.ylabel('largest eigenvalues of the covariance matrix of X');

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
#  ICA - Visualize kurtosis of components
#------------------------------------------------------------------------------
    # Plot attributes
    dictofdata = {'x': range(n_samples)}
    for i in range(n_features):
        feature = {i:X[:,i]}
        dictofdata.update(feature)
    PlotDataFrame(pd.DataFrame(dictofdata), "Attributes", dataset['name'], "instance", "value")

    print("ICA Visualize components")
    ica = FastICA()
    projected = ica.fit_transform(X)
    dictofdata = {'x': range(n_samples)}
    for i in range(n_features):
        feature = {i:projected[:,i]}
        dictofdata.update(feature)
    PlotDataFrame(pd.DataFrame(dictofdata), "ICA Components", dataset['name'], "sample", "value")

    ica = FastICA()
    S = ica.fit_transform(X)
    kurtosis = pd.DataFrame(S).kurt(axis=0).abs()

    plt.bar([i for i in range(1,len(kurtosis)+1)], kurtosis)
    plt.xlabel('component')
    plt.ylabel('kurtosis');

    import uuid
    plt.suptitle("ICA:"+dataset['name'])
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()

    dictofdata = {'x': range(n_samples)}
    for i in range(n_features):
        if kurtosis[i] == max(kurtosis):
            feature = {i:projected[:,i]}
            dictofdata.update(feature)
    PlotDataFrame(pd.DataFrame(dictofdata), "MAX Kurtosis", dataset['name'], "sample", "value")

    plotthis = list()
    for n_components in range(2,n_features):
        ica = FastICA(n_components=n_components)
        S = ica.fit_transform(X)
        plotthis.append(pd.DataFrame(S).kurt(axis=0).abs().mean())

    plt.plot(range(2,n_features),plotthis)
    plt.xlabel('components')
    plt.ylabel('kurtosis');
    plt.xticks(range(2,n_features))

    import uuid
    plt.suptitle("ICA:"+dataset['name'])
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()


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
