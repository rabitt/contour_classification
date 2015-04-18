""" Utilities for classifier experiments """
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def cross_val_sweep(X_train, Y_train, max_search=100, step=5, plot=True):
    """ Choose best parameter by performing cross fold validation

    Parameters
    ----------
    X_train : np.array [n_samples, n_features]
        Training features.
    Y_train : np.array [n_samples]
        Training labels
    max_search : int
        Maximum depth value to sweep
    step : int
        Step size in parameter sweep
    plot : bool
        If true, plot error bars and cv accuracy

    Returns
    -------
    best_depth : int
        Optimal max_depth parameter
    max_cv_accuracy : DataFrames
        Best accuracy achieved on hold out set with optimal parameter.
    """
    scores = []
    for max_depth in np.arange(5, max_search, step):
        print "training with max_depth=%s" % max_depth
        clf = RFC(n_estimators=100, max_depth=max_depth, n_jobs=-1, 
                  class_weight='auto')
        all_scores = cross_validation.cross_val_score(clf, X_train, Y_train, 
                                                      cv=5)
        scores.append([max_depth, np.mean(all_scores), np.std(all_scores)])

    depth = [score[0] for score in scores]
    accuracy = [score[1] for score in scores]
    std_dev = [score[2] for score in scores]

    if plot:
        plt.errorbar(depth, accuracy, std_dev, linestyle='-', marker='o')
        plt.title('Mean cross validation accuracy')
        plt.xlabel('max depth')
        plt.ylabel('mean accuracy')
        plt.show()

    best_depth = depth[np.argmax(accuracy)]
    max_cv_accuracy = np.max(accuracy)

    return best_depth, max_cv_accuracy


def train_clf(X_train, Y_train, best_depth):
    """ Train classifier.

    Parameters
    ----------
    X_train : np.array [n_samples, n_features]
        Training features.
    Y_train : np.array [n_samples]
        Training labels
    best_depth : int
        Optimal max_depth parameter

    Returns
    -------
    clf : classifier
        Trained scikit-learn classifier
    """
    clf = RFC(n_estimators=100, max_depth=best_depth, n_jobs=-1, 
              class_weight='auto')
    clf = clf.fit(X_train, Y_train)
    return clf


def clf_predictions(X_train, X_test, clf):
    """ Compute probability predictions for all training and test examples.

    Parameters
    ----------
    X_train : np.array [n_samples, n_features]
        Training features.
    X_test : np.array [n_samples, n_features]
        Testing features.
    clf : classifier
        Trained scikit-learn classifier

    Returns
    -------
    P_train : np.array [n_samples]
        predicted probabilities for training set
    P_test : np.array [n_samples]
        predicted probabilities for testing set
    """ 
    P_train = clf.predict_proba(X_train)
    P_test = clf.predict_proba(X_test)
    return P_train, P_test


def clf_metrics(P_train, P_test, Y_train, Y_test):
    """ Compute metrics on classifier predictions

    Parameters
    ----------
    P_train : np.array [n_samples]
        predicted probabilities for training set
    P_test : np.array [n_samples]
        predicted probabilities for testing set
    Y_train : np.array [n_samples]
        Training labels.
    Y_test : np.array [n_samples]
        Testing labels.

    Returns
    -------
    clf_scores : dict
        classifier scores for training set
    """ 
    Y_pred_train = 1*(P_train >= 0.5)
    Y_pred_test = 1*(P_test >= 0.5)

    train_scores = {}
    test_scores = {}

    train_scores['accuracy'] = metrics.accuracy_score(Y_train, Y_pred_train)
    test_scores['accuracy'] = metrics.accuracy_score(Y_test, Y_pred_test)

    train_scores['confusion matrix'] = \
        metrics.confusion_matrix(Y_train, Y_pred_train, labels=[0, 1])
    test_scores['confusion matrix'] = \
        metrics.confusion_matrix(Y_test, Y_pred_test, labels=[0, 1])

    train_scores['auc score'] = \
        metrics.roc_auc_score(Y_train, P_train + 1, average='weighted')
    test_scores['auc score'] = \
        metrics.roc_auc_score(Y_test, P_test + 1, average='weighted')

    clf_scores = {'train': train_scores, 'test': test_scores}

    return clf_scores

