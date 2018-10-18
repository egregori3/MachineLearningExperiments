# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/

import numpy as np 
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates
from pandas.plotting import radviz
import matplotlib.pyplot as plt

columns = []
columns.append('y1')
for i in range(1,17):
    columns.append('x'+str(i))
dataF=pd.read_csv("letter-recognition.data", sep=',',header=None, names=columns)
for x in columns:
    if x=='y1': continue
    print(x+" max: "+str(dataF[x].max()), end="    ")
    print("mean: "+str(dataF[x].mean()),  end="    ")
    print("min: "+str(dataF[x].min()))

if 1:
    dataF['y1'] = dataF['y1'].apply(lambda y: ord(y)-65)
    scatter_matrix(dataF, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

if 0:
    # Parallel Coordinates
    plt.figure()
    radviz(dataF,'y1')
    plt.show()

if 0:
    ser = dataF
    ser.plot.kde(subplots=True, layout=(2,12), legend=False, Label=False,yticks=[],xticks=[])
    plt.show()

