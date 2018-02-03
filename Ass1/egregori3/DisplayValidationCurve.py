import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

def DisplayValidationCurve(clf,X,y,validation_param,validation_param_range,title, kfolds):
    # -----------------------------------------------------------------------------
    # Validation Curve
    # -----------------------------------------------------------------------------
    train_scores, test_scores = validation_curve(
        clf, X, y, param_name=validation_param, param_range=validation_param_range,
        cv=kfolds, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(validation_param)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(validation_param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(validation_param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(validation_param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(validation_param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")

