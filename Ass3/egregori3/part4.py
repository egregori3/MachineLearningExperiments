from sklearn.preprocessing import scale
from ass1.PlotClassifiers import PlotClassifiers
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# components       PCA (95% variance) ICA (kurtosis) RP(10% error) LDA(95% variance)
scripts = {'wifi':       {'pca':4,        'ica':4,      'rp':6,        'lda':2 },
           'letter':     {'pca':10,       'ica':13,     'rp':14,       'lda':8 }}


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
    labels = dataset['y']
    script = scripts[dataset['name']]

    print("FULL NN")
    PlotClassifiers(X, labels, 'best', dataset['name']+":FULL", dataset['classes'])

    print("PCA NN")
    pca = PCA(n_components=script['pca'])
    projected = pca.fit_transform(X)
    PlotClassifiers(projected, labels, 'best', dataset['name']+":PCA", dataset['classes'])

    print("ICA NN")
    ica = FastICA(n_components=script['ica'])
    projected = ica.fit_transform(X)
    PlotClassifiers(projected, labels, 'best', dataset['name']+":ICA", dataset['classes'])

    print("RP NN")
    transformer = random_projection.GaussianRandomProjection(script['rp'])
    projected = transformer.fit_transform(X)
    PlotClassifiers(projected, labels, 'best', dataset['name']+":RP", dataset['classes'])

    print("LDA NN")
    transformer = LinearDiscriminantAnalysis(n_components=script['lda'])
    projected = transformer.fit_transform(X, labels)
    PlotClassifiers(projected, labels, 'best', dataset['name']+":LDA", dataset['classes'])
