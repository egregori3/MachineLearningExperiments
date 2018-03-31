

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------- PART 4 - Take part 2 results (one dataset) and run on ass1 NN --------
#-------------- Apply the dimensionality reduction algorithms ----------------- 
#-------------- to the two datasets and describe what you see. --------------- 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def part4( dataset ):
    print("PART 4 - "+dataset['name'])
    X = scale(dataset['X'])
    n_samples, n_features = X.shape
    labels = dataset['y']
    n_classes = len(np.unique(labels))
