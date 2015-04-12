import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import mir_eval

def load_contour_data(fpath):
    contour_data = pd.read_csv(fpath, header=None, index_col=None, delimiter=',')
    del contour_data[0]
    del contour_data[1]
    headers = contour_data.columns.values.astype('str')
    headers[0:12] = ['onset','offset','duration','pitch mean','pitch std',\
                    'salience mean', 'salience std', 'salience tot', \
                    'vibrato', 'vib rate', 'vib extent', 'vib coverage']
    contour_data.columns = headers
    return contour_data


def features_from_contour_data(contour_data):
    features = contour_data.iloc[:, 2:12]
    return features


def contours_from_contour_data(contour_data):
    contours = contour_data.iloc[:, 12:]
    contours_times = contours.iloc[:, 0::3]
    contours_freqs = contours.iloc[:, 1::3]
    contours_sal = contours.iloc[:, 2::3]
    return contours_times, contours_freqs, contours_sal

def load_annotation():
    pass

def make_coverage_plot(contours, annotation):
    pass

def label_contours(contours, annotation, olap_thresh):
    pass