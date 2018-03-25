# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html


from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture


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

    