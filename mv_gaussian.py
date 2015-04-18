""" Functions for doing scoring based on multivariate gaussian as in Meloida
"""
import numpy as np
from scipy.stats import boxcox
from scipy.stats import multivariate_normal
from sklearn import metrics


def transform_features(X_train, X_test):
    """ Transform features using a boxcox transform. Remove vibrato features.
    Comptes the optimal value of lambda on the training set and applies this
    lambda to the testing set.

    Parameters
    ----------
    X_train : np.array [n_samples, n_features]
        Untransformed training features.
    X_test : np.array [n_samples, n_features]
        Untransformed testing features.

    Returns
    -------
    X_train_boxcox : np.array [n_samples, n_features_trans]
        Transformed training features.
    X_test_boxcox : np.array [n_samples, n_features_trans]
        Transformed testing features.
    """
    X_train = X_train[:, 0:6]
    X_test = X_test[: 0:6]

    _, n_feats = X_train.shape

    X_train_boxcox = np.zeros(X_train.shape)
    lmbda_opt = np.zeros((n_feats,))

    eps = 1.0  # shift features away from zero
    for i in range(n_feats):
        X_train_boxcox[:, i], lmbda_opt[i] = boxcox(X_train[:, i] + eps)

    X_test_boxcox = np.zeros(X_test.shape)
    for i in range(n_feats):
        X_test_boxcox[:, i] = boxcox(X_test[:, i] + eps, lmbda=lmbda_opt[i])

    return X_train_boxcox, X_test_boxcox


def fit_gaussians(X_train_boxcox, Y_train):
    """ Fit class-dependent multivariate gaussians on the training set.

    Parameters
    ----------
    X_train_boxcox : np.array [n_samples, n_features_trans]
        Transformed training features.
    Y_train : np.array [n_samples]
        Training labels.

    Returns
    -------
    rv_pos : multivariate normal
        multivariate normal for melody class
    rv_neg : multivariate normal
        multivariate normal for non-melody class
    """    
    pos_idx = np.where(Y_train == 1)[0]
    mu_pos = np.mean(X_train_boxcox[pos_idx, :], axis=0)
    cov_pos = np.cov(X_train_boxcox[pos_idx, :], rowvar=0)

    neg_idx = np.where(Y_train == 0)[0]
    mu_neg = np.mean(X_train_boxcox[neg_idx, :], axis=0)
    cov_neg = np.cov(X_train_boxcox[neg_idx, :], rowvar=0)
    rv_pos = multivariate_normal(mean=mu_pos, cov=cov_pos, allow_singular=True)
    rv_neg = multivariate_normal(mean=mu_neg, cov=cov_neg, allow_singular=True)
    return rv_pos, rv_neg


def melodiness(sample, rv_pos, rv_neg):
    """ Compute melodiness score for an example given trained distributions.

    Parameters
    ----------
    sample : np.array [n_feats]
        Instance of transformed data.
    rv_pos : multivariate normal
        multivariate normal for melody class
    rv_neg : multivariate normal
        multivariate normal for non-melody class

    Returns
    -------
    melodiness: float
        score between 0 and inf. class cutoff at 1
    """       
    return rv_pos.pdf(sample)/rv_neg.pdf(sample)


def compute_all_melodiness(X_train_boxcox, X_test_boxcox, rv_pos, rv_neg):
    """ Compute melodiness for all training and test examples.

    Parameters
    ----------
    X_train_boxcox : np.array [n_samples, n_features_trans]
        Transformed training features.
    X_test_boxcox : np.array [n_samples, n_features_trans]
        Transformed testing features.
    rv_pos : multivariate normal
        multivariate normal for melody class
    rv_neg : multivariate normal
        multivariate normal for non-melody class

    Returns
    -------
    M_train : np.array [n_samples]
        melodiness scores for training set
    M_test : np.array [n_samples]
        melodiness scores for testing set
    """ 
    n_train = X_train_boxcox.shape[0]
    n_test = X_test_boxcox.shape[0]

    M_train = np.zeros((n_train, ))
    M_test = np.zeros((n_test, ))

    for i, sample in enumerate(X_train_boxcox):
        M_train[i] = melodiness(sample, rv_pos, rv_neg)

    for i, sample in enumerate(X_test_boxcox):
        M_test[i] = melodiness(sample, rv_pos, rv_neg)

    return M_train, M_test


def melodiness_metrics(M_train, M_test, Y_train, Y_test):
    """ Compute metrics on melodiness score

    Parameters
    ----------
    M_train : np.array [n_samples]
        melodiness scores for training set
    M_test : np.array [n_samples]
        melodiness scores for testing set
    Y_train : np.array [n_samples]
        Training labels.
    Y_test : np.array [n_samples]
        Testing labels.

    Returns
    -------
    melodiness_scores : dict
        melodiness scores for training set
    """ 
    M_bin_train = 1*(M_train >= 1)
    M_bin_test = 1*(M_test >= 1)

    train_scores = {}
    test_scores = {}

    train_scores['accuracy'] = metrics.accuracy_score(Y_train, M_bin_train)
    test_scores['accuracy'] = metrics.accuracy_score(Y_test, M_bin_test)

    train_scores['confusion matrix'] = \
        metrics.confusion_matrix(Y_train, M_bin_train, labels=[0, 1])
    test_scores['confusion matrix'] = \
        metrics.confusion_matrix(Y_test, M_bin_test, labels=[0, 1])

    train_scores['auc score'] = \
        metrics.roc_auc_score(Y_train, M_train + 1, average='weighted')
    test_scores['auc score'] = \
        metrics.roc_auc_score(Y_test, M_test + 1, average='weighted')

    melodiness_scores = {'train': train_scores, 'test': test_scores}

    return melodiness_scores

