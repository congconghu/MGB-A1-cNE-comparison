import time
import glob
import os

import numpy as np
from scipy import stats


class Timer:
    """Timer to get the time elapsed running a process"""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def get_A1_MGB_files(datafolder, stim):
    # get files for recordings in A1 and MGB
    files_all = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = {'A1': [x for x in files_all if 'H22x32' in x],
             'MGB': [x for x in files_all if 'H31x64' in x]}
    return files


def get_distance(position1, position2, direction='vertical'):

    if direction == 'vertical':
        dist = abs(position1[1] - position2[1])
    elif direction == 'horizontal':
        dist = abs(position1[0] - position2[0])
    elif direction == 'all':
        dist = ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
    return dist


def chi_square(x, n):
    """
    chi-square test for 2 ratios
    input
        x: number of observation of success
        n: sample size for the 2 samples
    """
    x = np.array(x)
    n = np.array(n)
    ratio = sum(x) / sum(n)
    observed = np.array([x[0], n[0] - x[0], x[1], n[1] - x[1]])
    x_null = n * ratio
    expected = np.array([x_null[0], n[0] - x_null[0], x_null[1], n[1] - x_null[1]])
    chi2stat = sum((observed - expected) ** 2 / expected)
    p = 1 - stats.chi2.cdf(chi2stat, 1)
    return chi2stat, p


