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
        plt.title("k-means statistics", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("clusters")
        plt.ylabel("stat")

        import uuid
        plt.suptitle(name)
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------- PART 3 - Run EM and K-means on part 2 results -----------------
#-------------- Apply the dimensionality reduction algorithms ----------------- 
#-------------- to the two datasets and describe what you see. --------------- 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part3( dataset ):
    print("PART 3 - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))
    pca2 = PCA(n_components=2)
    ica2 = FastICA(n_components=2)
    ra2  = random_projection.GaussianRandomProjection(2)
    lda2 = LinearDiscriminantAnalysis(n_components=2)

    VisualizeModel(KMeans(n_clusters=n_classes, init='k-means++').fit(pca2.fit_transform(X)), pca2.fit_transform(X), "PCA+K-Means")
    VisualizeModel(KMeans(n_clusters=n_classes, init='k-means++').fit(ica2.fit_transform(X)), ica2.fit_transform(X), "ICA+K-Means")
    VisualizeModel(KMeans(n_clusters=n_classes, init='k-means++').fit(ra2.fit_transform(X)), ra2.fit_transform(X), "RA+K-Means")
#    VisualizeModel(KMeans(n_clusters=n_classes, init='k-means++').fit(lda2.fit_transform(X)), lda2.fit_transform(X), "LDA+K-Means")

#------------------------------------------------------------------------------
#  Benchmark KM on reduced dataset
#------------------------------------------------------------------------------
    pca = PCA(n_components=n_classes)
    projected = pca.fit_transform(X)
    PlotKMStats(projected, labels, n_classes, "PCA: "+dataset['name'])


#------------------------------------------------------------------------------
#  PCA 2 dimension, k-means dataset clusters
#------------------------------------------------------------------------------
    print("PCA 2 dimensions, k-means %d clusters" % (n_classes) )
    reduced_data = pca2.fit_transform(X)
    VisualizeModel(model, reduced_data, title)


