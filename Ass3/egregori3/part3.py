


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------- PART 3 - Run EM and K-means on part 2 results -----------------
#-------------- Apply the dimensionality reduction algorithms ----------------- 
#-------------- to the two datasets and describe what you see. --------------- 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part3( dataset ):
    print("Part 2 PCA - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))
    pca2 = PCA(n_components=2)

#------------------------------------------------------------------------------
#  PCA 2 dimension, k-means dataset clusters
#------------------------------------------------------------------------------
    print("PCA 2 dimensions, k-means %d clusters" % (n_classes) )
    projected = pca.fit_transform(X)


