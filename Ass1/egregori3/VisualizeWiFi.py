# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/

import numpy as np 
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Load data into Numpy array
data = np.loadtxt(fname="wifi_localization.txt", delimiter="\t")
columns=['x1', 'x2', 'x3','x4','x5','x6','x7','y1']
dataF = pandas.DataFrame(data, columns=columns)
for x in columns:
    print(x+" max: "+str(dataF[x].max()), end="    ")
    print("mean: "+str(dataF[x].mean()),  end="    ")
    print("min: "+str(dataF[x].min()))

#now plot using pandas 
scatter_matrix(dataF, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()

