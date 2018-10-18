import numpy as np
from sklearn.preprocessing import scale
from ass1.PlotClassifiers import PlotClassifiers
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture


n_init=10
max_iter=300
random_state=0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------- PART 5 - Apply the clustering algorithms to the same dataset ---------
#---------- to which you just applied the dimensionality reduction ------------
#------------ treating the clusters as if they were new features. -------------
#------------------------------------------------------------------------------
#------------ In other words, treat the clustering algorithms as---------------
#--------------if they were dimensionality reduction algorithms. -------------- 
#---- Again, rerun your neural network learner on the newly projected data. ---
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part5( dataset ):
    print("PART 5 - "+dataset['name'])
    X = scale(dataset['X'])
    labels = dataset['y']
    n_classes = len(np.unique(labels))

    print("FULL NN")
    PlotClassifiers(X, labels, 'best', dataset['name']+":FULL", dataset['classes'])

    print("K-Means")
    km = KMeans(n_clusters=n_classes, init='k-means++', n_init=n_init, max_iter=max_iter, random_state=random_state)
    projected_km = km.fit(X)
    projected = np.column_stack([X,scale(projected_km.predict(X))])
    PlotClassifiers(projected, labels, 'best', dataset['name']+":kmeans", dataset['classes'])

    print("GM")
    gm_spherical = GaussianMixture(n_components=n_classes, covariance_type='spherical', max_iter=200, random_state=0)
    projected_gm = gm_spherical.fit(X)
    projected = np.column_stack([X,scale(projected_gm.predict(X))])
    PlotClassifiers(projected, labels, 'best', dataset['name']+":GM", dataset['classes'])

