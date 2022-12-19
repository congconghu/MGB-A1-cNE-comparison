# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:48:38 2022

@author: Congcong
"""
import re
import os
import glob
import pickle
import pandas as pd
import numpy as np

import plot_box as plots


# -------------------plot 5ms binned strfs of cNEs and member neurons----------

datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-5ms-strf'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_5ms_strf_ne_and_members(ne, figpath)

# -------------------plot cNE construction procedure --------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-construction'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

file = files[0]
for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx+1, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_ne_construction(ne, figpath)

# -----------------------plot stack of xcorr ----------------------------------
datafolder = 'E:\Congcong\Documents\data\comparison\data-summary'
xcorr = pd.read_json(os.path.join(datafolder, 'member_nonmember_pair_xcorr.json'))
xcorr['xcorr'] = xcorr['xcorr'].apply(lambda x: np.array(x))
plots.plot_xcorr(xcorr)
                
    