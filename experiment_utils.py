from contour_classification.ShuffleLabelsOut import ShuffleLabelsOut
import contour_classification.contour_utils as cc
import json
import numpy as np
import os

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
    annot_path = os.path.join(os.environ['MEDLEYDB_PATH'], 'Annotations', \
                                 'Melody_Annotations', mel_dir)

    contour_fname = "%s_%s" % (track, contour_suffix)
    contour_fpath = os.path.join(contours_path, contour_fname)
    annot_fname = "%s_%s" % (track, annot_suffix)
    annot_fpath = os.path.join(annot_path, annot_fname)
    
    cdat = cc.load_contour_data(contour_fpath)
    adat = cc.load_annotation(annot_fpath)

    return cdat, adat


def contour_probs(clf, feat_data, contour_data):
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
    contour_data['mel_prob'] = -1
    X, Y = pd_to_sklearn(feat_data)
    probs = clf.predict_proba(X)
    mel_probs = [p[1] for p in probs]
    contour_data['mel_prob'] = mel_probs
    return contour_data
