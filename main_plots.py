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
import matplotlib.pyplot as plt

import plot_box as plots


# -------------------plot 5ms binned strf of cNEs and member neurons----------
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

# ----------------------plot ICweight correlation and matching ICs for split cNEs-----------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/data-pkl'
figure_folder = r'/Users/hucongcong/Documents/UCSF/data/figure/cNE-split-corr'
files = glob.glob(datafolder + r'\*split.pkl', recursive=False)
for idx, file in enumerate(files):
    print('({}/{}) plot ICweight match for {}'.format(idx + 1, len(files), file))
    with open(file, 'rb') as f:
        ne_split = pickle.load(f)
    filename = re.findall('\d{6}_\d{6}.*', file)[0]
    figure_path = os.path.join(figure_folder, re.sub('pkl', 'jpg', filename))
    plots.plot_ne_split_ic_weight_corr(ne_split, figure_path)
    plots.plot_ne_split_ic_weight_match(ne_split, figure_path)

# -----------------------plot cNE properties comparison--------------------------------------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/data-pkl'
figure_folder = r'/Users/hucongcong/Documents/UCSF/data/figure/cNE-properties'
fig = plt.figure()
ax = fig.add_axes([.1, .1, .8, .8])
plots.num_ne_vs_num_neuron(ax, datafolder, 'dmr')