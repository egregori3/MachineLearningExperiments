----------------------------------------------------------
Code for this project was taken from the following sources
----------------------------------------------------------


scikit-learn tutorials
----------------------
http://scikit-learn.org/stable/user_guide.html#user-guide
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
https://www.youtube.com/playlist?list=PLonlF40eS6nynU5ayxghbz2QpDsUAyCVF
https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf
http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
https://github.com/rasbt/python-machine-learning-book
http://scikit-learn.org/dev/modules/preprocessing.html#preprocessing
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html


keras
-----
https://keras.io/

Visualization
-------------
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
http://www.saedsayad.com/docs/multivariate_visualization.pdf
https://pandas.pydata.org/pandas-docs/stable/visualization.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
http://www.cs.uml.edu/~phoffman/viz/explain.htm
https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
https://www.dataquest.io/blog/learning-curves-machine-learning/
http://scikit-learn.org/stable/modules/learning_curve.html
http://scikit-learn.org/stable/modules/generated/sklearn.learning_curve.learning_curve.html#sklearn.learning_curve
http://www.ritchieng.com/machinelearning-learning-curve/
http://scikit-learn.org/stable/modules/model_evaluation.html

Decision Tree Classifier
----------------------
http://scikit-learn.org/stable/modules/tree.html#tree
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.score
http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
http://scikit-learn.org/stable/modules/tree.html#tree
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
https://stackoverflow.com/questions/32210569/using-gridsearchcv-with-adaboost-and-decisiontreeclassifier

Support Vector Machines
-----------------------
http://scikit-learn.org/0.16/modules/svm.html#
http://scikit-learn.org/dev/modules/svm.html#kernel-functions


KNN
---
http://scikit-learn.org/0.16/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

ADABoost
--------
http://scikit-learn.org/0.16/modules/ensemble.html#adaboost



----------------------------------------------------------
Instructions for reproducing data
----------------------------------------------------------

Environment built using Anaconda: conda env create -f CondaEnv.yml
If there are any problems importing the CondaEnv.yml file.
Just install: scikitlearn, pandas

To create figure 1: python VisualizeWiFi.py
To create figures 2: python VisualizeLetter.py

To collect data from all classifiers: python RunMe.py


Directory Tree
--------------
egregori
    Data
        Letter
            GridSearch
            Manual
                Learning = Learning curves and confusion matrix
                Validation = Validation curves
        WiFi
            GridSearch
            Manual
                Learning = Learning curves and confusion matrix
                Validation = Validation curves

    Experiments = Code I found online and experimented with
    FailedDatasets = datasets I treid that were not interesting
    Plots
       Letter
            GridSearch
            Manual
                Learning = Learning curves and confusion matrix
                Validation = Validation curves
        WiFi
            GridSearch
            Manual
                Learning = Learning curves and confusion matrix
                Validation = Validation curves

    BoostingAnalysis.py = Parameters and script for running AdaBoost classifier
    DecisionTreeAnalysis.py = Parameters and script for running decision tree classifier
    KNNAnalysis.py = Parameters and script for running KNN classifier
    MLPAnalysis.py = Parameters and script for running ANN classifier
    SVMAnalysis.py = Parameters and script for running SVM classifier

    DisplayValidationCurve.py = Utility function
    FindBestParameters.py = Utility function
    LoadPreprocessDataset.py = Utility function
    PlotClassifiers.py = Utility function
    PlotConfusionMatrix.py = Utility function
    PlotLearningCurve.py = Script engine

    letter-recognition.data = letter dataset
    wifi_localization.txt = wifi dataset
