""" Utilities for classifier experiments """
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def cross_val_sweep(X_train, Y_train, plot=True):
    scores = []
    for max_depth in np.arange(5, 100, 5):
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



    