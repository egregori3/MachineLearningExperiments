# http://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/

import numpy as np 
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def Visualize(dataset):
    print("Generate Scatter Matrix - "+dataset['name'])
    data = dataset['data']
    cols = len(data[0])
    columns=['x'+str(i+1) for i in range(cols)]
    columns[-1] = 'y1'
    dataF = pandas.DataFrame(data, columns=columns)
    for x in columns:
        print(x+" max: "+str(dataF[x].max()), end="    ")
        print("mean: "+str(dataF[x].mean()),  end="    ")
        print("min: "+str(dataF[x].min()))

    #now plot using pandas 
    scatter_matrix(dataF, alpha=0.2, figsize=(6, 6), diagonal='kde')

    import uuid
 #   plt.tight_layout()
    plt.suptitle(dataset['name'])
    plt.savefig("./Plots/"+uuid.uuid4().hex)
    plt.close()


