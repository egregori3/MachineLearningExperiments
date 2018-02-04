"""
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------------------------------------------------------
# find best parameters
# -----------------------------------------------------------------------------
def FindBestParameters( clf, parms, cv, scores, name, X_train, y_train, X_test,y_test):
    for score in scores:
        print("Tuning hyper-parameters for %s" % score)
        print()

        bclf = GridSearchCV(clf, parms, cv=cv,
                           scoring=score)
        bclf.fit(X_train, y_train)

        print("Best parameters set found on development set "+name)
        print()
        print(bclf.best_params_)
        print()
        mts = bclf.cv_results_['mean_test_score']
        rts = bclf.cv_results_['rank_test_score']
        mft = bclf.cv_results_['mean_fit_time']
        mst = bclf.cv_results_['mean_score_time']
        for rts,mts,mft,mst,params in zip(rts,mts,mft,mst,bclf.cv_results_['params']):
            print("%d, %0.3f, %0.3f, %0.3f, %r" % (rts,mts,mft,mst,params))
        print()
        print("Detailed classification report "+name)
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, bclf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        return bclf.best_params_