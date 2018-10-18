from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture


_twoDplots = 1
_runKM     = 1
_runEM     = 1

# components       PCA (95% variance) ICA (kurtosis) RP(10% error) LDA(95% variance)
scripts = {'wifi':       {'pca':4,        'ica':4,      'rp':6,        'lda':2 },
           'letter':     {'pca':10,       'ica':13,     'rp':14,       'lda':8 }}



def VisualizeModel(model, reduced_data, title):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    import uuid
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


def PlotKMStats(X, groundtruth, n_classes, name):
        homo  = list()
        comp  = list()
        vmeas = list()
        ars   = list()
        ami   = list()
        sil   = list()
        for i in range(2, (n_classes*2)+2):
            km = KMeans(n_clusters=i, init='k-means++')
            km.fit(X)
            homo.append(metrics.homogeneity_score(groundtruth, km.labels_))
            comp.append(metrics.completeness_score(groundtruth, km.labels_))
            vmeas.append(metrics.v_measure_score(groundtruth, km.labels_))
            ars.append(metrics.adjusted_rand_score(groundtruth, km.labels_))
            ami.append(metrics.adjusted_mutual_info_score(groundtruth,  km.labels_))
            sil.append(metrics.silhouette_score(X, km.labels_, metric='euclidean', sample_size=len(groundtruth)))

        # Make a data frame
        df=pd.DataFrame({'x': range(1, (n_classes*2)+1), 
                        'homo': homo, 'comp': comp, 'vmeas': vmeas, 'ars': ars, 'ami': ami, 'sil': sil})
        PlotDataFrame(df, "k-means statistics", name, "clusters", "stat")


def PlotEMStats(X, groundtruth, n_classes, name):
        aic  = list()
        bic  = list()
        for i in range(2, (n_classes*2)+2):
                gm_spherical = GaussianMixture(n_components=i, covariance_type='spherical', max_iter=200, random_state=0)
                gm_spherical.fit(X)
                aic.append(gm_spherical.aic(X))
                bic.append(gm_spherical.bic(X))

        # Make a data frame
        df=pd.DataFrame({'x': range(1, (n_classes*2)+1), 'aic': aic})
        PlotDataFrame(df, "EM statistics", name, "clusters", "stat")
        # Make a data frame
        df=pd.DataFrame({'x': range(1, (n_classes*2)+1), 'bic': bic})
        PlotDataFrame(df, "EM statistics", name, "clusters", "stat")


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------- PART 3 - Run EM and K-means on part 2 results -----------------
#-------------- Apply the dimensionality reduction algorithms ----------------- 
#-------------- to the two datasets and describe what you see. --------------- 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part3( dataset ):
    print("PART 3 - "+dataset['name'])
    script = scripts[dataset['name']]
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))
    pca2 = PCA(n_components=2).fit_transform(X)
    ica2 = FastICA(n_components=2).fit_transform(X)
    ra2  = random_projection.GaussianRandomProjection(2).fit_transform(X)
    lda2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X,labels)

    if _twoDplots:
        print("2D Plots")
        for clusters in range(int(n_classes/4),n_classes+1):
            VisualizeModel(KMeans(n_clusters=clusters, init='k-means++').fit(pca2), pca2, dataset['name']+": PCA(2)+K-Means("+str(clusters)+")")
            VisualizeModel(KMeans(n_clusters=clusters, init='k-means++').fit(ica2), ica2, dataset['name']+": ICA(2)+K-Means("+str(clusters)+")")
            VisualizeModel(KMeans(n_clusters=clusters, init='k-means++').fit(ra2),  ra2,  dataset['name']+": RA(2)+K-Means("+str(clusters)+")")
            VisualizeModel(KMeans(n_clusters=clusters, init='k-means++').fit(lda2), lda2, dataset['name']+": LDA(2)+K-Means("+str(clusters)+")")

#------------------------------------------------------------------------------
#  Benchmark KM on reduced datasets from part 2
#------------------------------------------------------------------------------
    if _runKM:
        print("Plot KM stats")
        pca = PCA(n_components=script['pca'])
        projected = pca.fit_transform(X)
        PlotKMStats(projected, labels, n_classes, "PCA("+str(script['pca'])+"):"+dataset['name'])

        ica = FastICA(n_components=script['ica'])
        projected = ica.fit_transform(X)
        PlotKMStats(projected, labels, n_classes, "ICA("+str(script['ica'])+"):"+dataset['name'])

        transformer = random_projection.GaussianRandomProjection(script['rp'])
        projected = transformer.fit_transform(X)
        PlotKMStats(projected, labels, n_classes, "RP("+str(script['rp'])+"):"+dataset['name'])

        transformer = LinearDiscriminantAnalysis(n_components=script['lda'])
        projected = transformer.fit_transform(X, labels)
        PlotKMStats(projected, labels, n_classes, "LDA("+str(script['lda'])+"):"+dataset['name'])


#------------------------------------------------------------------------------
#  Benchmark EM on reduced datasets from part 2
#------------------------------------------------------------------------------
    if _runEM:
        print("Plot EM stats")
        pca = PCA(n_components=script['pca'])
        projected = pca.fit_transform(X)
        PlotEMStats(projected, labels, n_classes, "PCA("+str(script['pca'])+"):"+dataset['name'])

        ica = FastICA(n_components=script['ica'])
        projected = ica.fit_transform(X)
        PlotEMStats(projected, labels, n_classes, "ICA("+str(script['ica'])+"):"+dataset['name'])

        transformer = random_projection.GaussianRandomProjection(script['rp'])
        projected = transformer.fit_transform(X)
        PlotEMStats(projected, labels, n_classes, "RP("+str(script['rp'])+"):"+dataset['name'])

        transformer = LinearDiscriminantAnalysis(n_components=script['lda'])
        projected = transformer.fit_transform(X, labels)
        PlotEMStats(projected, labels, n_classes, "LDA("+str(script['lda'])+"):"+dataset['name'])
