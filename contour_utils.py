""" Utility functions for processing contours """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import mir_eval

def load_contour_data(fpath):
    """ Load contour data from vamp output csv file """
    contour_data = pd.read_csv(fpath, header=None, index_col=None, \
                               delimiter=',')
    del contour_data[0] # all zeros
    del contour_data[1] # just an unnecessary  index
    headers = contour_data.columns.values.astype('str')
    headers[0:12] = ['onset', 'offset', 'duration', 'pitch mean', 'pitch std',\
                    'salience mean', 'salience std', 'salience tot', \
                    'vibrato', 'vib rate', 'vib extent', 'vib coverage']
    contour_data.columns = headers
    return contour_data


def features_from_contour_data(contour_data):
    """ Get subset of columns corresponding to features """
    features = contour_data.iloc[:, 2:12]
    features['labels'] = -1 # all labels are unset
    return features


def contours_from_contour_data(contour_data):
    """ Get raw contour information from contour data """
    contours = contour_data.iloc[:, 12:]
    contours_times = contours.iloc[:, 0::3]
    contours_freqs = contours.iloc[:, 1::3]
    contours_sal = contours.iloc[:, 2::3]
    return contours_times, contours_freqs, contours_sal

def load_annotation(fpath):
    """ Load an annotation file into a pandas Series """
    annot_data = pd.read_csv(fpath, parse_dates=True, \
                             index_col=False, header=None)
    annot_data.columns = ['time', 'f0']

    # Add column with annotation values in cents
    annot_data['cents'] = 1200.0*np.log2(annot_data['f0']/55.0)

    return annot_data


def make_coverage_plot(contour_data, annot_data):
    """ Plot contours against annotation """
    c_times, c_freqs, _ = contours_from_contour_data(contour_data)
    plt.figure()
    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()):
        times = times[1].values
        freqs = freqs[1].values
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        plt.plot(times, freqs, '.r')
    plt.plot(annot_data['time'], annot_data['f0'], '.g')
    plt.show()


def label_contours(contour_data, annot_data, olap_thresh):
    """ Assign labels to contours based on annotation """
    c_times, c_freqs, _ = contours_from_contour_data(contour_data)
    features = features_from_contour_data(contour_data)

    for (times, freqs) in zip(c_times.iterrows(), c_freqs.iterrows()):
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values
        
        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        
        # get segment of ground truth matching this contour
        gt = annot_data[annot_data['time'] >= times[0]]
        gt = gt[gt['Time'] <= times[-1]]9
        
        # compute metrics
        res = mir_eval.melody.evaluate(gt['time'].values, gt['f0'].values, times, freqs)
        if res['Overall Accuracy'] >= olap_thresh:
            features.ix[row_idx, 'labels'] = 1
        else:
            features.ix[row_idx, 'labels'] = 0
            
    return features


def find_overlapping_contours(contour_data, annot_data, olap_thresh):
    olap_contours = data.copy()

    for (times, freqs) in zip(contours_times.iterrows(), contours_freqs.iterrows()):
        row_idx = times[0]
        times = times[1].values
        freqs = freqs[1].values
        
        # remove trailing NaNs
        times = times[~np.isnan(times)]
        freqs = freqs[~np.isnan(freqs)]
        
        # get segment of ground truth matching this contour
        gt = annot_data[annot_data['Time'] >= times[0]]
        gt = gt[gt['Time'] <= times[-1]]
        
        # compute metrics
        res = mir_eval.melody.evaluate(gt['Time'].values, gt['f0'].values, times, freqs)
        if res['Raw Pitch Accuracy'] == 0:
            olap_contours.drop(row_idx, inplace=True)
        if res['Overall Accuracy'] > 0.5:
            features.ix[row_idx, 'labels'] = 1
        if res['Overall Accuracy'] > 0 and res['Overall Accuracy'] <= 0.5:
            plt.figure
            plt.plot(gt['Time'], gt['f0'], '.g')
            plt.plot(times, freqs, '.r')


