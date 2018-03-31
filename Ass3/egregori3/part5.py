



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
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))
