from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------------------------------------------------------
# find best parameters
# -----------------------------------------------------------------------------
def FindBestParameters( clf, parms, cv, scores, name, X, y, test_size):
    # -----------------------------------------------------------------------------
    # Split dataset into training and test sets
    # -----------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_size, random_state = 0)

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
        if 0:
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in bclf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                    % (mean_score, scores.std() * 2, params))
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