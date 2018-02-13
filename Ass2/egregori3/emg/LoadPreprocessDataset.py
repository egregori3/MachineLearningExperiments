import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def LoadPreprocessDataset(argv, test_size=0.3):

    mydata = argv[1]
    # -----------------------------------------------------------------------------
    # Load dataset 
    # -----------------------------------------------------------------------------
    if not (mydata=='wifi' or mydata=='letter' or mydata=='wifi2' or mydata=='wifi3'):
        print("!ERROR! - Set dataset") 
        quit()

    if mydata == 'wifi':
        print("Loading WIFI dataset")
        dataset = np.loadtxt(".//DataSets//wifi_localization.txt", delimiter="\t")
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # -----------------------------------------------------------------------------
        X = dataset[:,0:7]
        y = dataset[:,7]
        print("X=")
        print(X)
        print("y=")
        print(y)
        name = 'WIFI'
        classes = ['1','2','3','4']


    if mydata == 'wifi2':
        print("Loading WIFI dataset")
        dataset = np.loadtxt(".//DataSets//wifi_localization.txt", delimiter="\t")
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # -----------------------------------------------------------------------------
        X = dataset[:,0:7]
        X = np.delete(X,1,1)
        X = np.delete(X,1,1)
        y = dataset[:,7]
        print("X=")
        print(X)
        print("y=")
        print(y)
        name = 'WIFI2'
        classes = ['1','2','3','4']

    if mydata == 'wifi3':
        print("Loading WIFI dataset")
        dataset = np.loadtxt(".//DataSets//wifi_localization.txt", delimiter="\t")
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # -----------------------------------------------------------------------------
        X = dataset[:,0:7]
        X = np.delete(X,6,1)
        y = dataset[:,7]
        print("X=")
        print(X)
        print("y=")
        print(y)
        name = 'WIFI3'
        classes = ['1','2','3','4']

    if mydata == 'robot':
        print("Loading ROBOT dataset")
        columns = []
        for i in range(1,25):
            columns.append('x'+str(i))
        columns.append('y1')
        df=pd.read_csv("..//DataSets/sensor_readings_24.data", sep=',',header=None, names=columns)
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # convert category stings into numeric values
        # -----------------------------------------------------------------------------
        print("Integer Encoding dataset")
        categories = {'Move-Forward':1, 'Sharp-Right-Turn':2, 'Slight-Right-Turn':3, 'Slight-Left-Turn':4}
        df['y1'] = df['y1'].apply(lambda y: categories[y])
        dataset = df.values
        X = dataset[:,0:24]
        y = dataset[:,24]
        print("X=")
        print(X)
        print("y=")
        print(y)
        name = 'ROBOT'
        classes = categories.keys()


    if mydata == 'letter':
        print("Loading LETTER dataset")
        columns = []
        columns.append('y1')
        for i in range(1,17):
            columns.append('x'+str(i))
        df=pd.read_csv(".//DataSets/letter-recognition.data", sep=',',header=None, names=columns)
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # convert category stings into numeric values
        # -----------------------------------------------------------------------------
        print("Integer Encoding dataset")
        df['y1'] = df['y1'].apply(lambda y: ord(y)-65)
        dataset = df.values
        X = dataset[:,1:18]
        y = dataset[:,0]
        print("X=")
        print(X)
        print("y=")
        print(y)
        name = 'LETTER'
        classes = [chr(c) for c in range(65,(65+26))]

    # -----------------------------------------------------------------------------
    # Print dataset characteristics
    # -----------------------------------------------------------------------------
    num_of_attributes = len(X[0])
    print("\n\nfirst:", end="\t")
    for i in range(0,num_of_attributes):
        print(X[0,i], end="\t")
    print("\nmin:", end="\t")
    for i in range(0,num_of_attributes):
        print(min( X[:,i]), end="\t")
    print("\nmax:", end="\t")
    for i in range(0,num_of_attributes):
        print(max( X[:,i]), end="\t")
    print("\nmean:", end="\t")
    for i in range(0,num_of_attributes):
        print(np.mean( X[:,i]), end="\t")
    print("\n")

    # -----------------------------------------------------------------------------
    # Scale data between -1.0 and 1.0
    # -----------------------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaledX = scaler.fit_transform(X)

    # -----------------------------------------------------------------------------
    # Split dataset into training and test sets
    # -----------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split( scaledX, y, test_size=test_size, random_state=0)


    return X_train, X_test, y_train, y_test, name, classes