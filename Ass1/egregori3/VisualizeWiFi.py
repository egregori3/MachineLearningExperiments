# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/

import numpy as np 
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Load data into Numpy array
data = np.loadtxt(fname="wifi_localization.txt", delimiter="\t")
dataF = pandas.DataFrame(data, columns=['x1', 'x2', 'x3','x4','x5','x6','x7','y1'])
print(dataF)

#now plot using pandas 
scatter_matrix(dataF, alpha=0.2, figsize=(6, 6), diagonal='hist')
plt.show()

