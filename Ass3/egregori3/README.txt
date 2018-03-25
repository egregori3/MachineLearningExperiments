## CS7641 Assignment 3 Eric Gregori

### Code Sources
https://github.com/rasbt/python-machine-learning-book-2nd-edition
https://github.com/rasbt/python-machine-learning-book/tree/master/code
Please review each code module for additional code sources

### pip freeze
certifi==2018.1.18
cycler==0.10.0
kiwisolver==1.0.1
matplotlib==2.2.0
numpy==1.14.1
pandas==0.22.0
pyparsing==2.2.0
python-dateutil==2.7.0
pytz==2018.3
scikit-learn==0.19.1
scipy==1.0.0
six==1.11.0
tornado==5.0
wincertstore==0.2

### Running the code
```
python assignment3.py
```

Each simulation is enabled or disabled by a if 0: or if 1:
Sub simulations can be enabled at the top of each module.

### Directory Structure
DataSets:                                           WiFi and letter datasets
python-machine-learning-book-2nd-edition-master:    Code examples from book
Assignment3.py                                      Run this
LoadPreprocessedDataset.py                          Loads datasets
Visualize.py                                        Create scatter matrix
part1.py                                            k-means and EM

### References
https://arxiv.org/ftp/arxiv/papers/1405/1405.7471.pdf
http://madhugnadig.com/articles/machine-learning/2017/03/04/implementing-k-means-clustering-from-scratch-in-python.html
https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
https://hub.packtpub.com/k-means-clustering-python/

https://github.com/rasbt/python-machine-learning-book
http://scikit-learn.org/stable/modules/clustering.html
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
