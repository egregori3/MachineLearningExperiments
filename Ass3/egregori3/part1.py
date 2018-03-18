# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
# http://scikit-learn.org/stable/modules/mixture.html#mixture
# https://github.com/rasbt/python-machine-learning-book-2nd-edition

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples


_statistics     = 0     # set to 1 to output statistics
_elbowplot      = 1     # Set to 1 to create elbow plot
_silhouetteplot = 1     # Set to 1 to create silhouette plots


def bench_k_means(estimator, name, X, labels, sample_size):
    t0 = time()
    estimator.fit(X)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(X, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


def part1( dataset ):
    print("k-means - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    n_classes = len(np.unique(dataset['y']))
    labels = dataset['y']

    if _statistics: 
        sample_size = 300
        print("n_classes: %d, \t n_samples %d, \t n_features %d"
            % (n_classes, n_samples, n_features))

        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

        bench_k_means(KMeans(init='k-means++', n_clusters=n_classes, n_init=10),
                             name="k-means++", data=data, labels=labels, sample_size=sample_size)

        bench_k_means(KMeans(init='random', n_clusters=n_classes, n_init=10),
                             name="random", data=data, labels=labels, sample_size=sample_size)

    if _elbowplot: 
        distortions = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, 
                        init='k-means++', 
                        n_init=10, 
                        max_iter=300, 
                        random_state=0)
            km.fit(X)
            distortions.append(km.inertia_)
        plt.plot(range(1, 11), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()

        import uuid
        plt.suptitle(dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()

    if _silhouetteplot: 
        km = KMeans(n_clusters=3, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        y_km = km.fit_predict(X)

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
        plt.suptitle(dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()


