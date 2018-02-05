from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PlotLearningCurve import PlotLearningCurve
from LoadPreprocessDataset import LoadPreprocessDataset
from FindBestParameters import FindBestParameters
from DisplayValidationCurve import DisplayValidationCurve
from PlotConfusionMatrix import PlotConfusionMatrix
from PlotROCCrossval import PlotROCCrossval

def PlotClassifiers(list_of_dicts,CreateClassifier,clf,kfolds,test_size,prefix,scores,lop,tuned_parameters,manual_params):
    import matplotlib.pyplot as plt
    dataset = ""
    test_size = test_size
    for params in list_of_dicts:
        if 'testsize' in params:
            test_size = params['testsize']
            print("Setting test_size to "+str(test_size))

        if 'dataset' in params and params['dataset'] != dataset:
            # load dataset 
            NotUsed,y,name,sX,classes = LoadPreprocessDataset([0,params['dataset']])
            dataset = params['dataset']
            # -----------------------------------------------------------------------------
            # Split dataset into training and test sets
            # -----------------------------------------------------------------------------
            X_train, X_test, y_train, y_test = train_test_split( sX, y, test_size=test_size, random_state=0)

        if 'default' in params and params['default'] == 'best':
            best_params = FindBestParameters(   clf, 
                                                tuned_parameters, 
                                                kfolds, 
                                                scores, 
                                                name,
                                                X_train,y_train,
                                                X_test,y_test
                                            )
            top = best_params.copy()

        if 'default' in params and params['default'] == 'manual':
            top = manual_params.copy()

        for parameter in lop:
            if parameter in params.keys():
                top[parameter] = params[parameter]
        clf = CreateClassifier( top )

        # -----------------------------------------------------------------------------
        # Generate classification report
        # ----------------------------------------------------------------------------- 
        print("Classification report")
        print("Parameters:")
        print(top)
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

        # -----------------------------------------------------------------------------
        # Put parameters in title
        # ----------------------------------------------------------------------------- 
        pvalues = ""
        add_lf = 50
        for pname in lop:
            pvalues += (pname+":"+str(top[pname])+",")
            if len(pvalues) > add_lf:
                pvalues += "\n"
                add_lf += 50

        # -----------------------------------------------------------------------------
        # Generate plots
        # ----------------------------------------------------------------------------- 
        for plottype in params['type']:
            if len(plottype.split('='))>1:
                plottype,scorer = plottype.split('=')
            # -----------------------------------------------------------------------------
            # Confusion Matrix
            # ----------------------------------------------------------------------------- 
            if plottype == 'CM':
                title = name+" "+prefix+" Confusion Matrix"+"\n"+pvalues
                plt.figure()
                PlotConfusionMatrix(clf,X_train,y_train,X_test,y_test,classes,title=title)

            # -----------------------------------------------------------------------------
            # Learning Curve
            # -----------------------------------------------------------------------------
            if plottype == 'LC':
                title = name+" "+prefix+" Learning Curve "+scorer+"\n"+pvalues
                plt = PlotLearningCurve(clf, title, X_train,y_train, cv=kfolds, scorer=scorer)

            # -----------------------------------------------------------------------------
            # Validation Curve
            # -----------------------------------------------------------------------------
            if plottype == 'VC':
                title = name+" "+prefix+" Validation Curve"+"\n"+pvalues
                plt.figure()
                DisplayValidationCurve(clf, X_train,y_train, params['vc_name'], params['vc_range'], title, kfolds)

    plt.show()
