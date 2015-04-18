""" Functions for doing scoring based on multivariate gaussian as in Meloida
"""
import numpy as np
from scipy.stats import boxcox
from scipy.stats import multivariate_normal

def transform_features(X_train, X_test):

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
    pos_idx = np.where(Y_train == 1)[0]
    mu_pos = np.mean(X_train_boxcox[pos_idx, :], axis=0)
    cov_pos = np.cov(X_train_boxcox[pos_idx, :], rowvar=0)

    neg_idx = np.where(Y_train == 0)[0]
    mu_neg = np.mean(X_train_boxcox[neg_idx, :], axis=0)
    cov_neg = np.cov(X_train_boxcox[neg_idx, :], rowvar=0)
    rv_pos = multivariate_normal(mean=mu_pos, cov=cov_pos, allow_singular=True)
    rv_neg = multivariate_normal(mean=mu_neg, cov=cov_neg, allow_singular=True)
    return rv_pos, rv_neg

def melodiness(x, rv_pos, rv_neg):
    return rv_pos.pdf(x)/rv_neg.pdf(x)