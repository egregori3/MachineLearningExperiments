import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def LoadPreprocessDataset(argv):

    mydata = argv[1]
    # -----------------------------------------------------------------------------
    # Load dataset 
    # -----------------------------------------------------------------------------
    if not (mydata=='wifi' or mydata=='letter'):
        print("!ERROR! - Set dataset") 
        quit()

    if mydata == 'wifi':
        print("Loading WIFI dataset")
        dataset = np.loadtxt("wifi_localization.txt", delimiter="\t")
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # -----------------------------------------------------------------------------
        X = dataset[:,0:7]
        y = dataset[:,7]
        print(y)
        name = 'WIFI'
        classes = ['1','2','3','4']

    if mydata == 'robot':
        print("Loading ROBOT dataset")
        columns = []
        for i in range(1,25):
            columns.append('x'+str(i))
        columns.append('y1')
        df=pd.read_csv("sensor_readings_24.data", sep=',',header=None, names=columns)
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
        print(y)
        name = 'ROBOT'
        classes = categories.keys()

    if mydata == 'letter':
        print("Loading LETTER dataset")
        columns = []
        columns.append('y1')
        for i in range(1,17):
            columns.append('x'+str(i))
        df=pd.read_csv("letter-recognition.data", sep=',',header=None, names=columns)
        # -----------------------------------------------------------------------------
        # Preprocess dataset into (X)input and (y)output
        # convert category stings into numeric values
        # -----------------------------------------------------------------------------
        print("Integer Encoding dataset")
        df['y1'] = df['y1'].apply(lambda y: ord(y)-65)
        dataset = df.values
        X = dataset[:,1:18]
        y = dataset[:,0]
        print(y)
        name = 'LETTER'
        classes = [char(c) for c in range(65,(65+26))]

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    sX = scaler.fit_transform(X)

    return X,y,name,sX,classes