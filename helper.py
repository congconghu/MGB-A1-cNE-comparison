import time
import glob
import os

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


