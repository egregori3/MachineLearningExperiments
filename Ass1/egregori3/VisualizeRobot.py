# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/

import numpy as np 
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

columns = []
for i in range(1,25):
	columns.append('x'+str(i))
columns.append('y1')
print(columns)
df=pd.read_csv("sensor_readings_24.data", sep=',',header=None, names=columns)
print(df)

# Convert categories from strings into numeric values
categories = {'Move-Forward':1, 'Sharp-Right-Turn':2, 'Slight-Right-Turn':3, 'Slight-Left-Turn':4}
df['y1'] = df['y1'].apply(lambda y: categories[y])
print(df)

#now plot using pandas 
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()

