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

# ++++++++++++++++++++++++++++++++++++++++ single units properties +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------------------------- plot strf of all units------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-summary'
units = pd.read_json(os.path.join(datafolder, 'single_units.json'))
figfolder = r'E:\Congcong\Documents\data\comparison\figure\su-strf'
plots.plot_strf_df(units, figfolder, order='strf_ri_z', properties=True, smooth=True)
figfolder = r'E:\Congcong\Documents\data\comparison\figure\su-crh'
plots.plot_crh_df(units, figfolder, order='crh_ri_z', properties=True)
figfolder = r'E:\Congcong\Documents\data\comparison\figure\su-strf\nonlinearity'
plots.plot_strf_nonlinearity_df(units, figfolder)

# +++++++++++++++++++++++++++++++++++++++++++ cNE properties ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------ plot 5ms binned strf of cNEs and member neurons-----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-5ms-strf'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_5ms_strf_ne_and_members(ne, figpath)

# ------------------------------ plot crh of cNEs and member neurons-----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-crh'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_crh_ne_and_members(ne, figpath)

# --------------------------------plot cNE construction procedure -----------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-construction'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx+1, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_ne_construction(ne, figpath)

# --------------------------------plot cNE properties comparison--------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figure_folder = r'/Users/hucongcong/Documents/UCSF/data/figure/cNE-properties'
fig = plt.figure()
ax = fig.add_axes([.1, .1, .8, .8])
for stim in ('dmr', 'spon'):
    # scatter plot of number of cNE vs number of neurons
    plots.num_ne_vs_num_neuron(ax, datafolder, stim=stim)
    fig.savefig(os.path.join(figure_folder, 'n_ne_vs_n_neuron_{}.jpg'.format(stim)))
    ax.clear()
    # scatter plot of cNE size vs number of neurons
    plots.ne_size_vs_num_neuron(ax, datafolder, stim=stim, plot_type='mean')
    fig.savefig(os.path.join(figure_folder, 'ne_size_vs_n_neuron_mean_{}.jpg'.format(stim)))
    ax.clear()
    # scatter plot of frequncy range ne vs all neurons
    plots.ne_freq_span_vs_all_freq_span(ax, datafolder, stim)
    fig.savefig(os.path.join(figure_folder, 'freq_span_ne_vs_all_{}.jpg'.format(stim)))
    ax.clear()
    # box plot of cNE size vs number of neurons
    plots.ne_size_vs_num_neuron(ax, datafolder, stim=stim, plot_type='raw', relative=True)
    fig.savefig(os.path.join(figure_folder, 'ne_size_ratio_{}.jpg'.format(stim)))
    ax.clear()
    # box plot of cNE size vs number of neurons with recordings of certain sizes
    plots.ne_size_vs_num_neuron(ax, datafolder, stim=stim, plot_type='raw', relative=True, n_neuron_filter=(13, 29))
    fig.savefig(os.path.join(figure_folder, 'ne_size_ratio_sub_{}.jpg'.format(stim)))
    ax.clear()
    # bar plot of number of cNEs each neuron participated in
    plots.num_ne_participate(ax, datafolder, stim=stim)
    fig.savefig(os.path.join(figure_folder, 'n_participation_{}.jpg'.format(stim)))
    ax.clear()
    # bar plot of number of cNEs each neuron participated in with recordings of certain sizes
    plots.num_ne_participate(ax, datafolder, stim=stim, n_neuron_filter=(13, 29))
    fig.savefig(os.path.join(figure_folder, 'n_participation_sub_{}.jpg'.format(stim)))
    ax.clear()
    # vertical spacial distribution
    for probe in ('H31x64', 'H22x32'):
        # pairwise distance
        plots.ne_member_distance(ax, datafolder, stim=stim, probe=probe, direction='vertical')
        fig.savefig(os.path.join(figure_folder, 'member_distance_{}_{}.jpg'.format(probe, stim)))
        ax.clear()
        # cNE span
        plots.ne_member_span(ax, datafolder, stim=stim, probe=probe)
        fig.savefig(os.path.join(figure_folder, 'member_span_{}_{}.jpg'.format(probe, stim)))
        ax.clear()
        # pairwise frequency difference
        plots.ne_member_freq_distance(ax, datafolder, stim=stim, probe=probe)
        fig.savefig(os.path.join(figure_folder, 'member_freq_distance_{}_{}.jpg'.format(probe, stim)))
        ax.clear()
        # cNE freq span
        plots.ne_member_freq_span(ax, datafolder, stim=stim, probe=probe)
        fig.savefig(os.path.join(figure_folder, 'member_freq_span_{}_{}.jpg'.format(probe, stim)))
        ax.clear()
    # horizontal distribution
    # pairwise distance
    plots.ne_member_distance(ax, datafolder, stim=stim, probe='H22x32', direction='horizontal')
    fig.savefig(os.path.join(figure_folder, 'member_distance_shank_{}.jpg'.format(stim)))
    ax.clear()
    # cNE span
    plots.ne_member_shank_span(ax, datafolder, stim=stim, probe='H22x32')
    fig.savefig(os.path.join(figure_folder, 'member_span_shank_{}.jpg'.format(stim)))
    ax.clear()

# ---------------------------------------------plot stack of xcorr ----------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-summary'
xcorr = pd.read_json(os.path.join(datafolder, 'member_nonmember_pair_xcorr.json'))
xcorr['xcorr'] = xcorr['xcorr'].apply(lambda x: np.array(x))
plots.plot_xcorr(xcorr)


# +++++++++++++++++++++++++++++++++++++++ split cNE: spon/dmr stability +++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------------plot ICweight correlation and matching ICs for split cNEs-----------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/data-pkl'
figure_folder = r'/Users/hucongcong/Documents/UCSF/data/figure/cNE-split-corr'
files = glob.glob(datafolder + r'/*split.pkl', recursive=False)
for idx, file in enumerate(files):
    print('({}/{}) plot ICweight match for {}'.format(idx + 1, len(files), file))
    with open(file, 'rb') as f:
        ne_split = pickle.load(f)
    filename = re.findall('\d{6}_\d{6}.*', file)[0]
    figure_path = os.path.join(figure_folder, re.sub('pkl', 'jpg', filename))
    plots.plot_ne_split_ic_weight_corr(ne_split, figure_path)
    plots.plot_ne_split_ic_weight_match(ne_split, figure_path)

# --------------------------------------plot ICweight correlation null distribution-------------------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/data-pkl'
figure_folder = r'/Users/hucongcong/Documents/UCSF/data/figure/cNE-split-corr-null'
files = glob.glob(datafolder + r'/*split.pkl', recursive=False)
for idx, file in enumerate(files):
    print('({}/{}) plot ICweight match for {}'.format(idx + 1, len(files), file))
    with open(file, 'rb') as f:
        ne_split = pickle.load(f)
    filename = re.findall('\d{6}_\d{6}.*', file)[0]
    figure_path = os.path.join(figure_folder, re.sub('pkl', 'jpg', filename))
    plots.plot_ne_split_ic_weight_null_corr(ne_split, figure_path)


# +++++++++++++++++++++++++++++++++++++++ cNE stimulus encoding +++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------percentage of stimulus encoding units------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
savefolder = r'E:\Congcong\Documents\data\comparison\figure\cNE-properties'
# plots.plot_ne_neuron_stim_response_hist(datafolder, savefolder)
plots.plot_ne_neuron_strf_response_hist(datafolder, savefolder)

# ------------------------------ plot strf, crh and nonlinearity of cNEs and member neurons------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-strf-crh-nonlinearity'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_ne_member_strf_crh_nonlinearity(ne, figpath)
    
# ------------------------------ plot strf, crh and nonlinearity of subsampled spikes------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-summary\subsample'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-strf-crh-nonlinearity-subsampled'
files = glob.glob(datafolder + r'\*subsample_ri.json.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_ne_member_strf_crh_nonlinearity_subsample(ne, figpath)
    
    