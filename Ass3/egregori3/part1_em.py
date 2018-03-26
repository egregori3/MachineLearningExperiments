# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
# http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture


from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples
from PlotConfusionMatrix import PlotConfusionMatrix
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture


_statistics      = 1     # set to 1 to output statistics
_confusionmatrix = 1     # set to 1 to output confusion matrix
_silhouetteplot  = 1     # Set to 1 to create silhouette plots


def bench_em(estimator, name, X, labels, sample_size):
    t0 = time()
    estimator.fit(X)
    predict_labels = estimator.predict(X)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), 
             metrics.homogeneity_score(labels, predict_labels ),
             metrics.completeness_score(labels, predict_labels ),
             metrics.v_measure_score(labels, predict_labels ),
             metrics.adjusted_rand_score(labels, predict_labels ),
             metrics.adjusted_mutual_info_score(labels,  predict_labels ),
             metrics.silhouette_score(X, predict_labels ,
                                      metric='euclidean',
                                      sample_size=sample_size)),
            estimator.aic(X), estimator.bic(X)
          )


def ConfusionMatrix(algo, X, title, labels, classes):
    data = algo.fit(X)
    predict_labels = data.predict(X)
    # Scale dataset labels to start with 0
    dataset_labels_starting_at_zero = [i-min(labels) for i in labels]
    print("Dataset - labels: "+str(np.unique(labels)))
    print("Kmeans - labels: "+str(np.unique(predict_labels)))
    PlotConfusionMatrix( dataset_labels_starting_at_zero, predict_labels, classes, title="EM:"+title)

    import uuid
    plt.tight_layout()
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()


def sp(algo, X, title):
    y_km = algo.fit(X)
    y_km = y_km.predict(X)
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
        
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()

    import uuid
    plt.suptitle("EM:"+title)
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---------------  PART 1 - Run EM and K-means on two datasets------------------
#---------- You can choose your own measures of distance/similarity.----------- 
#--------------- Naturally, you'll have to justify your choices, -------------- 
#------------- but you're practiced at that sort of thing by now. -------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part1_em( dataset ):
    print("k-means - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))

    print("n_classes: %d, \t n_samples %d, \t n_features %d" % (n_classes, n_samples, n_features))
    print("labels: "+str(np.unique(labels)))

#------------------------------------------------------------------------------
#  Expectation Maximization using GausianMixture models
#------------------------------------------------------------------------------
    gm_spherical = GaussianMixture(n_components=n_classes, covariance_type='spherical', max_iter=200, random_state=0)
    gm_diag = GaussianMixture(n_components=n_classes, covariance_type='diag', max_iter=200, random_state=0)
    gm_tied = GaussianMixture(n_components=n_classes, covariance_type='tied', max_iter=200, random_state=0)
    gm_full = GaussianMixture(n_components=n_classes, covariance_type='full', max_iter=200, random_state=0)

#------------------------------------------------------------------------------
#  Statistics
#------------------------------------------------------------------------------
    if _statistics:
        print(82 * '_')
        print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette\tAIC\tBIC')
        bench_em(gm_spherical, name="gm_spherical", X=X, labels=labels, sample_size=len(labels))
        bench_em(gm_diag, name="gm_diag", X=X, labels=labels, sample_size=len(labels))
        bench_em(gm_tied, name="gm_tied", X=X, labels=labels, sample_size=len(labels))
        bench_em(gm_full, name="gm_full", X=X, labels=labels, sample_size=len(labels))

#------------------------------------------------------------------------------
#  Confusion Matrix
#------------------------------------------------------------------------------
    if _confusionmatrix:
        ConfusionMatrix(gm_spherical, X, "spherical", labels, dataset['classes'])
        ConfusionMatrix(gm_diag, X, "diag", labels, dataset['classes'])
        ConfusionMatrix(gm_tied, X, "tied", labels, dataset['classes'])
        ConfusionMatrix(gm_full, X, "full", labels, dataset['classes'])

#------------------------------------------------------------------------------
#  Silhoutteplot
#------------------------------------------------------------------------------
    if _silhouetteplot: 
        sp(gm_spherical, X, "spherical")
        sp(gm_diag, X, "diag")
        sp(gm_tied, X, "tied")
        sp(gm_full, X, "full")
