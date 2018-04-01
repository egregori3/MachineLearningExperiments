# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://etav.github.io/python/scikit_pca.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
# https://piazza.com/class/jc2dhebfn0n2qo?cid=952
# http://scikit-learn.org/stable/modules/decomposition.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.kurtosis.html
# https://github.com/JonathanTay/CS-7641-assignment-3


from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv


_twoDVisualize  = 0
_calcComponents = 0
_ScatterMatrix  = 1

# components       PCA (95% variance) ICA (kurtosis) RP(10% error) LDA(95% variance)
scripts = {'wifi':       {'pca':4,        'ica':4,      'rp':6,        'lda':2 },
           'letter':     {'pca':10,       'ica':13,     'rp':14,       'lda':8 }}


# https://github.com/JonathanTay/CS-7641-assignment-3
def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)


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


def PlotScatter(X, y, chart_title):
    for attribute in range(len(X[0])):
        plt.scatter(X[:,attribute],y,
                    c=y, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('rainbow', 10))
        plt.xlabel("X"+str(attribute))
        plt.ylabel('labels')
        plt.colorbar();

        import uuid
        plt.suptitle(chart_title)
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()


def ScatterMatrix(X, y, title):
    print("Generate Scatter Matrix - "+title)
    cols = len(X[0])
    columns=['x'+str(i+1) for i in range(cols)]
    dataF = pd.DataFrame(X, columns=columns)
    dataF['y1'] = y
    for x in columns:
        print(x+" max: "+str(dataF[x].max()), end="    ")
        print("mean: "+str(dataF[x].mean()),  end="    ")
        print("min: "+str(dataF[x].min()))

    #now plot using pandas 
    scatter_matrix(dataF, alpha=0.2, figsize=(6, 6), diagonal='kde')

    import uuid
 #   plt.tight_layout()
    plt.suptitle(title)
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
    script = scripts[dataset['name']]
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))

#------------------------------------------------------------------------------
#  PCA - Visualize data
#------------------------------------------------------------------------------
    if _twoDVisualize:
        print("PCA Visualization")
        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)
        Plot2d(projected, "PCA:"+dataset['name'], labels)

#------------------------------------------------------------------------------
#  PCA - Determeine best number of components
#------------------------------------------------------------------------------
    if _calcComponents: 
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
#  PCA - Scatter
#------------------------------------------------------------------------------
    if _ScatterMatrix:
        print("PCA ScatterMatrix")
        pca = PCA(n_components=script['pca'])
        projected = pca.fit_transform(X)
        ScatterMatrix(projected, labels, "PCA:"+dataset['name'])
        PlotScatter(projected, labels, "PCA:"+dataset['name'])

#------------------------------------------------------------------------------
#  ICA - Visualize data
#------------------------------------------------------------------------------
    if _twoDVisualize:
        print("ICA Visualization")
        ica = FastICA(n_components=2)
        projected = ica.fit_transform(X)
        Plot2d(projected, "ICA:"+dataset['name'], labels)

#------------------------------------------------------------------------------
#  ICA - Visualize kurtosis of components
#------------------------------------------------------------------------------
    if _calcComponents: 
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
#  ICA - Scatter
#------------------------------------------------------------------------------
    if _ScatterMatrix:
        print("ICA ScatterMatrix")
        ica = FastICA(n_components=script['ica'])
        projected = pca.fit_transform(X)
        ScatterMatrix(projected, labels, "ICA:"+dataset['name'])
        PlotScatter(projected, labels, "ICA:"+dataset['name'])


#------------------------------------------------------------------------------
#  Random Projection - Visualize data
#------------------------------------------------------------------------------
    if _twoDVisualize:
        print("Random Projection Visualization")
        transformer = random_projection.GaussianRandomProjection(2)
        projected = transformer.fit_transform(X)
        Plot2d(projected, "RP:"+dataset['name'], labels)

    if _calcComponents: 
        for _ in range(10):
            plotthis = list()
            for n_components in range(2,n_features):
                transformer = random_projection.GaussianRandomProjection(n_components=n_components)
                projected = transformer.fit(X)
                plotthis.append(reconstructionError(projected,X))

            plt.plot(range(2,n_features),plotthis)

        plt.xlabel('components')
        plt.ylabel('reconstruction error');
        plt.xticks(range(2,n_features))

        import uuid
        plt.suptitle("Random Projection:"+dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()

#------------------------------------------------------------------------------
#  RP - Scatter
#------------------------------------------------------------------------------
    if _ScatterMatrix:
        print("RP ScatterMatrix")
        rp = random_projection.GaussianRandomProjection(n_components=script['rp'])
        projected = rp.fit_transform(X)
        ScatterMatrix(projected, labels, "RP:"+dataset['name'])
        PlotScatter(projected, labels, "RP:"+dataset['name'])


#------------------------------------------------------------------------------
#  Linear Discriminant Analysis- Visualize data
#------------------------------------------------------------------------------
    if _twoDVisualize:
        print("Linear Discriminant Analysis Visualization")
        transformer = LinearDiscriminantAnalysis(n_components=2)
        projected = transformer.fit_transform(X, labels)
        Plot2d(projected, "LDA:"+dataset['name'], labels)

    if _calcComponents: 
        print("LDA Plot components versus variance")
        lda = LinearDiscriminantAnalysis().fit(X, labels)
        plt.plot(np.cumsum(lda.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');

        import uuid
        plt.suptitle("LDA:"+dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()

#------------------------------------------------------------------------------
#  LDA - Scatter
#------------------------------------------------------------------------------
    if _ScatterMatrix:
        print("LDA ScatterMatrix")
        lda = LinearDiscriminantAnalysis(n_components=script['lda'])
        projected = lda.fit_transform(X,labels)
        ScatterMatrix(projected, labels, "LDA:"+dataset['name'])
        PlotScatter(projected, labels, "LDA:"+dataset['name'])

