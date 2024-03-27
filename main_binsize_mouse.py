# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:46:36 2024

@author: Congcong
"""
import glob
import os
import pickle
import session_toolbox as mtp
import numpy as np
import re
import ne_toolbox as netools
import matplotlib.pyplot as plt
import plot_box as plots

# save data in pickle file
datafolder = r'E:\Congcong\Documents\data\cne_ftc\data2'
folders = glob.glob(datafolder + r'\*', recursive=False)

for idx, folder in enumerate(folders):
    print('({}/{})Save pickle file:'.format(idx+1, len(folders)), folder)
    file = os.path.join(folder, 'cne_data_actx.mat')
    session = mtp.Session()
    session.read_mat_file_from_cne_ftc(file)
    session.save_pkl_file(os.path.join(r'E:\Congcong\Documents\data\cne_ftc\data2',
                                          f'{session.exp}.pkl'))

# get 0.5ms binned spktrains
files = glob.glob(datafolder + r'\*.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    # save spktrains
    if not hasattr(session, 'spktrain'):
        print('({}/{}) Save spktrain for {}'.format(idx+1, len(files), file))
        session.save_spktrain()
    
# get cnes
files = glob.glob(datafolder + r'\*.pkl', recursive=False)
dfs = np.array([2, 5, 10, 20, 40, 80, 160])*2
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    # cNE analysis
    print('({}/{}) Get cNEs for {}'.format(idx+1, len(files), file))
    for df in dfs:
        print(df/2)
        savefile_path = re.sub(r'.pkl', '-ne-{}dft.pkl'.format(df), session.file_path)
        ne = session.get_ne(df=df, stim=None)
        ne.get_sham_patterns(nshift=1000)
        ne.save_pkl_file(savefile_path)

# match cNE patterns between different bin sizes
datafolder = r'E:\Congcong\Documents\data\cne_ftc\data2\binsize'
savefolder = r'E:\Congcong\Documents\data\cne_ftc\data2\ic_match_tbins'
dfs = np.array([2, 5, 10, 20, 40, 80, 160])*2
for df in dfs:
    files = glob.glob(datafolder + r'\*-{}dft.pkl'.format(df), recursive=False)
    for idx, file in enumerate(files):
        savefile = re.sub('.pkl', '-ic_match_tbins.pkl',os.path.basename(file))
        print(r'{}/{} Matching time bins for {}'.format(idx+1, len(files), savefile))
        ic_matched = netools.ICweight_match_binsize(datafolder, file, dfs)
        with open(os.path.join(savefolder, savefile), 'wb') as f:
            pickle.dump(ic_matched, f)
            
# data summary saved to dataframe   
datafolder = r'E:\Congcong\Documents\data\cne_ftc\data2\binsize'
savefolder = r'E:\Congcong\Documents\data\cne_ftc\data2\summary'
dfs = np.array([2, 5, 10, 20, 40, 80, 160])*2
netools.batch_save_icweight_binsize_corr_to_dataframe(datafolder, savefolder, dfs, stim=None)         


# ---------------------------------plots------------------------------------------
figfolder =  r'E:\Congcong\Documents\data\cne_ftc\data2\figure'
summaryfolder=r'E:\Congcong\Documents\data\cne_ftc\data2\summary'
datafolder = r'E:\Congcong\Documents\data\cne_ftc\data2\binsize'
# get total number of cNEs under each binsize:
# match cNE patterns between different bin sizes
dfs = np.array([2, 5, 10, 20, 40, 80, 160])*2
n_total = np.zeros(dfs.shape)
for i, df in enumerate(dfs):
    files = glob.glob(datafolder + r'\*-{}dft.pkl'.format(df), recursive=False)
    for idx, file in enumerate(files):
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        n_total[i] += ne.patterns.shape[0]
print(n_total)
            
# summary plot of matched bin sizes
fig = plt.figure(figsize=[3, 2])
ax = fig.add_axes([.1, .16, .8, .8])
im = plots.plot_binsizes_matching_prc(ax, summaryfolder=summaryfolder, n_total=n_total, p_thresh=.0001)
fig.colorbar(im, ax=ax)
fig.savefig(os.path.join(figfolder, 'heatmap_sig_match.pdf'), dpi=300) 
            
# correlation and shared member 5ms vs 160ms
fig = plt.figure(figsize=[3, 2])
# correlation values
x_start = .12
y_start = .12
x_fig = .3
y_fig = .8
x_space = .2
df = [20, 320]
for param in ('corr', 'member_overlap_prc_all'):
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plots.plot_icweight_corr_vs_reference_bin(ax, [20, 320], summary_folder=summaryfolder,
                                              stim=None, param=param, p_thresh=.0001)
    x_start += x_fig + x_space
fig.savefig(os.path.join(figfolder, 'hist_corr_member_share.pdf'), dpi=300) 
            