""" Module for generating melody output based on classifier scores """
import pandas as pd
import contour_classification.contour_utils as cc
import numpy as np
import mir_eval


def melody_from_clf(contour_data, prob_thresh=0.5):
    """ Compute output melody using classifier output.

    Parameters
    ----------
    contour_data : DataFrame or dict of DataFrames
        DataFrame containing labeled features.
    prob_thresh : float
        Threshold that determines positive class

    Returns
    -------
    mel_output : Series
        Pandas Series with time stamp as index and f0 as values
    """

    # remove contours below probability threshold
    contour_candidates = contour_data[contour_data['mel prob'] >= prob_thresh]
    probs = contour_candidates['mel prob']

    if len(contour_candidates) == 0:
        print "Warning: no contours above threshold."
        return None

    # get separate DataFrames of contour time, frequency, and probability
    contour_times, contour_freqs, _ = \
        cc.contours_from_contour_data(contour_candidates, n_end=4)
    contour_probs = pd.concat([probs]*contour_times.shape[1], axis=1,
                              ignore_index=True)

    # create DataFrame with all unwrapped [time, frequency, probability] values.
    mel_dat = pd.DataFrame(columns=['time', 'f0', 'probability'])
    mel_dat['time'] = contour_times.values.ravel()
    mel_dat['f0'] = contour_freqs.values.ravel()
    mel_dat['probability'] = contour_probs.values.ravel()

    # remove rows with NaNs
    mel_dat.dropna(inplace=True)

    # sort by probability then by time
    # duplicate times with have maximum probability value at the end
    mel_dat.sort(columns='probability', inplace=True)
    mel_dat.sort(columns='time', inplace=True)

    # compute evenly spaced time grid for output
    step_size = 128.0/44100.0  # contour time stamp step size
    mel_time_idx = np.arange(0, np.max(mel_dat['time'].values) + 1, step_size)

    # find index in evenly spaced grid of estimated time values
    old_times = mel_dat['time'].values
    reidx = np.searchsorted(mel_time_idx, old_times)
    shift_idx = (np.abs(old_times - mel_time_idx[reidx - 1]) < \
                 np.abs(old_times - mel_time_idx[reidx]))
    reidx[shift_idx] = reidx[shift_idx] - 1

    # remove duplicate time values
    mel_dat['reidx'] = reidx
    mel_dat.drop_duplicates(subset='reidx', take_last=True, inplace=True)

    #
    mel_output = pd.Series(np.zeros(mel_time_idx.shape), index=mel_time_idx)
    mel_output.iloc[mel_dat['reidx']] = mel_dat['f0'].values

    return mel_output


def score_melodies(mel_output_dict, test_annot_dict):
    """ Score melody output against ground truth.

    Parameters
    ----------
    mel_output_dict : dict of Series
        Dictionary of melody output series keyed by trackid
    test_annot_dict : dict of DataFrames
        Dictionary of DataFrames containing annotations.

    Returns
    -------
    melody_scores : dict
        melody evaluation metrics for each track
    """
    melody_scores = {}
    print "Scoring..."
    for key in mel_output_dict.keys():
        print key
        if mel_output_dict[key] is None:
            print "skipping..."
            continue
        ref = test_annot_dict[key]
        est = mel_output_dict[key]
        melody_scores[key] = mir_eval.melody.evaluate(ref['time'].values,
                                                      ref['f0'].values,
                                                      np.array(est.index),
                                                      est.values)

    return melody_scores


