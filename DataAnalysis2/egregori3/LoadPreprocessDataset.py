import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def LoadPreprocessDataset():

    datasets = {}
    # -----------------------------------------------------------------------------
    # Load wifi dataset 
    # -----------------------------------------------------------------------------
    print("Loading WIFI dataset")
    dataset = np.loadtxt("./DataSets/wifi_localization.txt", delimiter="\t")
    # -----------------------------------------------------------------------------
    # Preprocess dataset into (X)input and (y)output
    # -----------------------------------------------------------------------------
    X = dataset[:,0:7]
    y = dataset[:,7]
    classes = ['1','2','3','4']
    datasets['wifi'] = {'name':'wifi','data':dataset, 'X':X, 'y':y, 'classes':classes}


    # -----------------------------------------------------------------------------
    # Load letter dataset 
    # -----------------------------------------------------------------------------
    print("Loading LETTER dataset")
    columns = []
    columns.append('y1')
    for i in range(1,17):
        columns.append('x'+str(i))
    df=pd.read_csv("./DataSets/letter-recognition.data", sep=',',header=None, names=columns)
    # -----------------------------------------------------------------------------
    # Preprocess dataset into (X)input and (y)output
    # convert category stings into numeric values
    # -----------------------------------------------------------------------------
    print("Integer Encoding dataset")
    df['y1'] = df['y1'].apply(lambda y: ord(y)-65)
    dataset = df.values
    X = dataset[:,1:18]
    y = dataset[:,0]
    data = np.column_stack([X,y])
    classes = [chr(c) for c in range(65,(65+26))]
    datasets['letter'] = {'name':'letter','data':data, 'X':X, 'y':y, 'classes':classes}

    return datasets