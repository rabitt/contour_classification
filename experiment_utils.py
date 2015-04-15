from contour_classification.ShuffleLabelsOut import ShuffleLabelsOut
import contour_classification.contour_utils as cc
import json
import numpy as np
import os

def create_splits():
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

