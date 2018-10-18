# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
# http://scikit-learn.org/stable/modules/mixture.html#mixture
# https://github.com/rasbt/python-machine-learning-book-2nd-edition
# http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
# http://scikit-learn.org/stable/modules/clustering.html


from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples
from PlotConfusionMatrix import PlotConfusionMatrix


_statistics      = 1     # set to 1 to output statistics
_confusionmatrix = 1     # set to 1 to output confusion matrix
_elbowplot       = 1     # Set to 1 to create elbow plot
_silhouetteplot  = 1     # Set to 1 to create silhouette plots
_statsplot       = 1     # Set to 1 to create statistics plot

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


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---------------  PART 1 - Run EM and K-means on two datasets------------------
#---------- You can choose your own measures of distance/similarity.----------- 
#--------------- Naturally, you'll have to justify your choices, -------------- 
#------------- but you're practiced at that sort of thing by now. -------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part1_km( dataset ):
    print("Part 1 KM - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))

    print("n_classes: %d, \t n_samples %d, \t n_features %d" % (n_classes, n_samples, n_features))
    print("labels: "+str(np.unique(labels)))

#------------------------------------------------------------------------------
#  k-means
#------------------------------------------------------------------------------
    n_init=10
    max_iter=300
    random_state=0

    km_pp = KMeans(n_clusters=n_classes, init='k-means++', n_init=n_init, max_iter=max_iter, random_state=random_state)
    km_r  = KMeans(n_clusters=n_classes, init='random', n_init=n_init, max_iter=max_iter, random_state=random_state)

#------------------------------------------------------------------------------
#  Statistics
#------------------------------------------------------------------------------
    if _statistics: 
        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

        bench_k_means(km_pp, name="k-means++", X=X, labels=labels, sample_size=len(labels))
        bench_k_means(km_r, name="random", X=X, labels=labels, sample_size=len(labels))

#------------------------------------------------------------------------------
#  Confusion Matrix
#------------------------------------------------------------------------------
    if _confusionmatrix:
        data = km_pp.fit(X)
        # Scale dataset labels to start with 0
        dataset_labels_starting_at_zero = [i-min(labels) for i in labels]
        print("Dataset - labels: "+str(np.unique(labels)))
        print("Kmeans - labels: "+str(np.unique(data.labels_)))
        PlotConfusionMatrix( dataset_labels_starting_at_zero, data.labels_, dataset['classes'], title="KM")

        import uuid
        plt.tight_layout()
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()

#------------------------------------------------------------------------------
#  Elbowplot - Try k-means versus accuracy
#------------------------------------------------------------------------------
    if _elbowplot: 
        distortions = []
        for i in range(1, (n_classes*2)+1):
            km = KMeans(n_clusters=i, 
                        init='k-means++', 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        random_state=random_state)
            km.fit(X)
            distortions.append(km.inertia_)
        plt.plot(range(1, (n_classes*2)+1), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()

        import uuid
        plt.suptitle(dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()


#------------------------------------------------------------------------------
#  Silhoutteplot
#------------------------------------------------------------------------------
    if _silhouetteplot: 
        y_km = km_pp.fit_predict(X)
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
        plt.suptitle("KM:"+dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()

#------------------------------------------------------------------------------
#  Plot statistics for various clusters
#------------------------------------------------------------------------------
    if _statsplot: 
        homo  = list()
        comp  = list()
        vmeas = list()
        ars   = list()
        ami   = list()
        sil   = list()
        for i in range(2, (n_classes*2)+2):
            km = KMeans(n_clusters=i, 
                        init='k-means++', 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        random_state=random_state)
            km.fit(X)
            homo.append(metrics.homogeneity_score(labels, km.labels_))
            comp.append(metrics.completeness_score(labels, km.labels_))
            vmeas.append(metrics.v_measure_score(labels, km.labels_))
            ars.append(metrics.adjusted_rand_score(labels, km.labels_))
            ami.append(metrics.adjusted_mutual_info_score(labels,  km.labels_))
            sil.append(metrics.silhouette_score(X, km.labels_, metric='euclidean', sample_size=len(labels)))

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
        plt.suptitle(dataset['name'])
        plt.savefig("./Plots/"+uuid.uuid4().hex)
        plt.close()









