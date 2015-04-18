""" Helper functions for experiments """

from contour_classification.ShuffleLabelsOut import ShuffleLabelsOut
import contour_classification.contour_utils as cc
import json
import numpy as np
import os
import sys


def create_splits():
    """ Split MedleyDB into train/test splits.

    Returns
    -------
    mdb_files : list
        List of sorted medleydb files.
    splitter : iterator
        iterator of train/test indices.
    """
    index = json.load(open('medley_artist_index.json'))

    mdb_files = []
    keys = []

    for k, v in sorted(index.items()):
        mdb_files.append(k)
        keys.append(v)

    keys = np.asarray(keys)
    mdb_files = np.asarray(mdb_files)
    splitter = ShuffleLabelsOut(keys, random_state=1)

    # for train, test in splitter:
    #     train_tracks = mdb_files[train]
    #     test_tracks = mdb_files[test]
    #     break

    return mdb_files, splitter


def get_data_files(track, meltype=1):
    """ Load all necessary data for a given track and melody type.

    Parameters
    ----------
    track : str
        Track identifier.
    meltype : int
        Melody annotation type. One of [1, 2, 3]

    Returns
    -------
    cdat : DataFrame
        Pandas DataFrame of contour data.
    adat : DataFrame
        Pandas DataFrame of annotation data.
    """
    contour_suffix = \
        "MIX_vamp_melodia-contours_melodia-contours_contoursall.csv"
    contours_path = "melodia_contours"
    annot_suffix = "MELODY%s.csv" % str(meltype)
    mel_dir = "MELODY%s" % str(meltype)
    annot_path = os.path.join(os.environ['MEDLEYDB_PATH'], 'Annotations',
                              'Melody_Annotations', mel_dir)

    contour_fname = "%s_%s" % (track, contour_suffix)
    contour_fpath = os.path.join(contours_path, contour_fname)
    annot_fname = "%s_%s" % (track, annot_suffix)
    annot_fpath = os.path.join(annot_path, annot_fname)

    cdat = cc.load_contour_data(contour_fpath, normalize=True)
    adat = cc.load_annotation(annot_fpath)

    return cdat, adat


def compute_all_overlaps(train_tracks, test_tracks, meltype):
    """ Compute each contour's overlap with annotation.

    Parameters
    ----------
    train_tracks : list
        List of trackids in training set
    test_tracks : list
        List of trackids in test set
    meltype : int
        One of [1,2,3]

    Returns
    -------
    train_contour_list : list of DataFrames
        List of train feature data frames
    test_contour_list : list of DataFrames
        List of test feature data frames
    """

    train_contour_list = []

    msg = "Generating training features..."
    train_len = len(train_tracks)
    num_spaces = train_len - len(msg)
    print msg + ' '*num_spaces + '|'

    for track in train_tracks:
        cdat, adat = get_data_files(track, meltype=meltype)
        train_contour_list.append(cc.compute_overlap(cdat, adat))
        sys.stdout.write('.')

    print "-"*30

    test_contour_list = []

    msg = "Generating testing features..."
    test_len = len(test_tracks)
    num_spaces = test_len - len(msg)
    print msg + ' '*num_spaces + '|'

    for track in test_tracks:
        cdat, adat = get_data_files(track, meltype=meltype)
        test_contour_list.append(cc.compute_overlap(cdat, adat))
        sys.stdout.write('.')

    return train_contour_list, test_contour_list


def label_all_contours(train_feat_list, test_feat_list, olap_thresh):
    """ Add labels to features based on overlap_thresh.

    Parameters
    ----------
    train_feat_list : list of DataFrames
        List of train feature data frames
    test_feat_list : list of DataFrames
        List of test feature data frames
    olap_thresh : float
        Value in [0, 1). Min overlap to be labeled as melody.

    Returns
    -------
    train_feat_list : list of DataFrames
        List of train feature data frames
    test_feat_list : list of DataFrames
        List of test feature data frames
    """
    for i, train_feat in enumerate(train_feat_list):
        train_feat_list[i] = cc.label_contours(train_feat,
                                               olap_thresh=olap_thresh)

    for i, test_feat in enumerate(test_feat_list):
        test_feat_list[i] = cc.label_contours(test_feat,
                                              olap_thresh=olap_thresh)
    return train_feat_list, test_feat_list


def contour_probs(clf, contour_data):
    """ Compute classifier probabilities for contours.

    Parameters
    ----------
    clf : scikit-learn classifier
        Binary classifier.
    feat_data : DataFrame
        DataFrame with features.
    contour_data : DataFrame
        DataFrame with contour information.

    Returns
    -------
    all_features : DataFrame
        Merged feature data.
    """
    contour_data['mel prob'] = -1
    X, _ = cc.pd_to_sklearn(contour_data)
    probs = clf.predict_proba(X)
    mel_probs = [p[1] for p in probs]
    contour_data['mel prob'] = mel_probs
    return contour_data