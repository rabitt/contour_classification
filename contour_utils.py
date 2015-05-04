""" Utility functions for processing contours """

import pandas as pd
import numpy as np
import mir_eval
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_contour_data(fpath, normalize=True):
    """ Load contour data from vamp output csv file.
    Initializes DataFrame to have all future columns.

    Parameters
    ----------
    fpath : str
        Path to vamp output csv file.

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    """
    contour_data = pd.read_csv(fpath, header=None, index_col=None,
                               delimiter=',')
    del contour_data[0]  # all zeros
    del contour_data[1]  # just an unnecessary  index
    headers = contour_data.columns.values.astype('str')
    headers[0:12] = ['onset', 'offset', 'duration', 'pitch mean', 'pitch std',
                     'salience mean', 'salience std', 'salience tot',
                     'vibrato', 'vib rate', 'vib extent', 'vib coverage']
    contour_data.columns = headers
    contour_data.num_end_cols = 0
    contour_data['overlap'] = -1  # overlaps are unset
    contour_data['labels'] = -1  # all labels are unset
    contour_data['melodiness'] = ""
    contour_data['mel prob'] = -1
    contour_data.num_end_cols = 4

    if normalize:
        contour_data = normalize_features(contour_data)

    return contour_data


def normalize_features(contour_data):
    """ Normalizes (trackwise) features in contour_data.
    Adds labels column with all labels unset.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    normalize : Bool
        If true, performs trackwise normalization over salience.

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with normalized contour feature data.
    """

    _, _, contour_sal = contours_from_contour_data(contour_data)

    # maximum salience value across all contours
    sal_max = contour_sal.max().max()

    # normalize salience features by max salience
    contour_data['salience mean'] = contour_data['salience mean']/sal_max
    contour_data['salience std'] = contour_data['salience std']/sal_max

    # normalize saience total by max salience and duration
    contour_data['salience tot'] = \
        contour_data['salience tot']/(sal_max*contour_data['duration'])

    # compute min and max duration
    dur_min = contour_data['duration'].min()
    dur_max = contour_data['duration'].max()

    # normalize duration to be between 0 and 1
    contour_data['duration'] = \
        (contour_data['duration'] - dur_min)/(dur_max - dur_min)

    # give standardized duration back to total salience
    contour_data['salience tot'] = \
        contour_data['salience tot']*contour_data['duration']

    return contour_data


def contours_from_contour_data(contour_data, n_end=4):
    """ Get raw contour information from contour data

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.

    Returns
    -------
    contour_times : DataFrame
        Pandas data frame with all raw contour times.
    contour_freqs : DataFrame
        Pandas data frame with all raw contour frequencies (Hz).
    contour_sal : DataFrame
        Pandas data frame with all raw contour salience values.
    """
    contour_times = contour_data.iloc[:, 12:-n_end:3]
    contour_freqs = contour_data.iloc[:, 13:-n_end:3]
    contour_sal = contour_data.iloc[:, 14:-n_end:3]

    return contour_times, contour_freqs, contour_sal


def load_annotation(fpath):
    """ Load an annotation file into a pandas Series.
    Add column with frequency values also converted to cents.

    Parameters
    ----------
    fpath : str
        Path to annotation file.

    Returns
    -------
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    """
    annot_data = pd.read_csv(fpath, parse_dates=True,
                             index_col=False, header=None)
    annot_data.columns = ['time', 'f0']

    # Add column with annotation values in cents
    annot_data['cents'] = 1200.0*np.log2(annot_data['f0']/55.0)

    return annot_data


def plot_contours(contour_data, annot_data, contour_data2=None):
    """ Plot contours against annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    """
    if contour_data2 is not None:
        c_times2, c_freqs2, _ = contours_from_contour_data(contour_data2)
        for (times, freqs) in zip(c_times2.iterrows(), c_freqs2.iterrows()):
            times = times[1].values
            freqs = freqs[1].values
            times = times[~np.isnan(times)]
            freqs = freqs[~np.isnan(freqs)]
            plt.plot(times, freqs, '.c')

    c_times, c_freqs, _ = contours_from_contour_data(contour_data)
    plt.figure()
    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()):
        times = times[1].values
        freqs = freqs[1].values
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        plt.plot(times, freqs, '.r')

    plt.plot(annot_data['time'], annot_data['f0'], '.k')
    plt.show()


def compute_overlap(contour_data, annot_data):
    """ Compute percentage of overlap of each contour with annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.

    Returns
    -------
    feature_data : DataFrame
        Pandas data frame with feature_data and labels.
    """
    c_times, c_freqs, _ = contours_from_contour_data(contour_data)

    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()):
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values

        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]

        # get segment of ground truth matching this contour
        gt_segment = annot_data[annot_data['time'] >= times[0]]
        gt_segment = gt_segment[gt_segment['time'] <= times[-1]]

        # compute metrics
        res = mir_eval.melody.evaluate(gt_segment['time'].values,
                                       gt_segment['f0'].values, times, freqs)

        contour_data.ix[row_idx, 'overlap'] = res['Overall Accuracy']

    return contour_data


def label_contours(contour_data, olap_thresh):
    """ Compute contours based on annotation.
    Contours with at least olap_thresh overlap with annotation
    are labeled as positive examples. Otherwise negative.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.
    olap_thresh : float
        Overlap threshold for positive examples

    Returns
    -------
    contour_data : DataFrame
        Pandas data frame with contour_data and labels.
    """
    contour_data['labels'] = 1*(contour_data['overlap'] > olap_thresh)
    return contour_data


def contour_glass_ceiling(contour_data, annot_data):
    """ Get subset of contour data that overlaps with annotation.

    Parameters
    ----------
    contour_data : DataFrame
        Pandas data frame with all contour data.
    annot_data : DataFrame
        Pandas data frame with all annotation data.

    Returns
    -------
    olap_contours : DataFrame
        Subset of contour_data that overlaps with annotation.
    """
    orig_annot_data = annot_data.copy()

    contour_times, contour_freqs, _ = contours_from_contour_data(contour_data)

    mel_dat = pd.DataFrame(columns=['time', 'f0', 'annot_dist'])
    mel_dat['time'] = contour_times.values.ravel()
    mel_dat['f0'] = contour_freqs.values.ravel()
    mel_dat['annot_dist'] = -1

    mel_dat.dropna(inplace=True)
    mel_dat = mel_dat[mel_dat['f0'] != 0]

    annot_times = annot_data['time'].values
    old_times = mel_dat['time'].values
    reidx = np.searchsorted(annot_times, old_times)
    reidx[reidx >= len(annot_times)] = len(annot_times) - 1 

    shift_idx = (np.abs(old_times - annot_times[reidx - 1]) < \
                 np.abs(old_times - annot_times[reidx]))
    reidx[shift_idx] = reidx[shift_idx] - 1
    mel_dat.index = reidx
    mel_dat['reidx'] = reidx

    non_zero_annot = np.intersect1d(annot_data[annot_data['f0'] != 0].index, 
                                    mel_dat.index)
    mel_dat = mel_dat.loc[non_zero_annot, :]

    mel_dat['annot_dist'] = np.abs(mel_dat['f0'] - 
                                   annot_data.loc[mel_dat.index, 'f0'])

    mel_dat.sort(columns='annot_dist', inplace=True)
    mel_dat.sort(columns='time', inplace=True)
    mel_dat.drop_duplicates(subset='reidx', take_last=False, inplace=True)

    #############################
    step_size = 128.0/44100.0  # contour time stamp step size
    mel_time_idx = np.arange(0, np.max(mel_dat['time'].values) + 1, step_size)

    # find index in evenly spaced grid of estimated time values
    old_times = mel_dat['time'].values
    reidx = np.searchsorted(mel_time_idx, old_times)
    shift_idx = (np.abs(old_times - mel_time_idx[reidx - 1]) < \
                 np.abs(old_times - mel_time_idx[reidx]))
    reidx[shift_idx] = reidx[shift_idx] - 1

    # find duplicate time values
    mel_dat['reidx2'] = reidx

    mel_dat.drop_duplicates(subset='reidx', take_last=True, inplace=True)

    mel_output = pd.Series(np.zeros(mel_time_idx.shape), index=mel_time_idx)
    mel_output.iloc[mel_dat['reidx2']] = mel_dat['f0'].values

    res = mir_eval.melody.evaluate(orig_annot_data['time'].values, 
                                   orig_annot_data['f0'].values,
                                   np.array(mel_output.index), 
                                   mel_output.values)

    return res, mel_output


def join_contours(contours_list):
    """ Merge features for a multiple track into a single DataFrame

    Parameters
    ----------
    contours_list : list of DataFrames
        List of Pandas data frames with labeled features.

    Returns
    -------
    all_contours : DataFrame
        Merged feature data.
    """
    all_contours = pd.concat(contours_list, ignore_index=False)
    return all_contours


def pd_to_sklearn(contour_data):
    """ Convert pandas data frame to sklearn style features and labels

    Parameters
    ----------
    contour_data : DataFrame or dict of DataFrames
        DataFrame containing labeled features.

    Returns
    -------
    features : np.ndarray
        fetures (n_samples x n_features)
    labels : np.1darray
        Labels (n_samples,)
    """

    #  Reduce before join for speed and memory saving
    if isinstance(contour_data, dict):
        red_list = []
        lab_list = []

        for key in contour_data.keys():
            red_list.append(contour_data[key].iloc[:, 2:12])
            lab_list.append(contour_data[key]['labels'])

        joined_data = join_contours(red_list)
        joined_labels = join_contours(lab_list)

    else:
        joined_data = contour_data.iloc[:, 2:12]
        joined_labels = contour_data['labels']

    features = np.array(joined_data)
    labels = np.array(joined_labels)

    return features, labels

