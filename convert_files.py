import os
import glob
import pickle
import re
import mmap

import session_toolbox as mtp
import helper
from scipy.io import loadmat

import pandas as pd
import numpy as np

# ------------------------------------------- pickle single units-------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-Matlab'
files = glob.glob(datafolder + r'\*split.mat', recursive=False)

for idx, file in enumerate(files):
    print('({}/{})Save pickle file:'.format(idx+1, len(files)), file)
    session = mtp.Session()
    session.read_mat_file(file)
    file_pkl = re.sub('-spk-curated-split.mat', '.pkl', file)
    file_pkl = re.sub('Matlab', 'pkl', file_pkl)
    session.save_pickle_file(file_pkl)

# load pickle file
with open(file_pkl, 'rb') as f:
    session = pickle.load(f)

# ------------------------------------------- pickle stimulus----------------------------------------------------------
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_stim.mat'
stimfile_pkl = re.sub('_stim.mat', '.pkl', stimfile)

stimfile_path = os.path.join(stimfolder, stimfile)
stimfile_pkl_path = os.path.join(stimfolder, stimfile_pkl)

# save stimulus files from mat to pickle
# takes ~20s to load mat file
# takes ~4s to load pickle file
stim = mtp.Stimulus()
with helper.Timer():
    stim.read_mat_file(stimfile_path)
stim.save_pkl_file(stimfile_pkl_path)

# load stimulus files from pickle file
with helper.Timer():
    with open(stimfile_pkl_path, 'rb') as f:
        stim = pickle.load(f)

# get sta bigmat
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim = pickle.load(f)
stim.down_sample(10)
nleads = 20
stim_mat = stim.stim_mat
bigmat = np.empty([stim_mat.shape[0] * nleads, stim_mat.shape[1]])
with helper.Timer():
    for i in range(nleads, stim_mat.shape[1]+1):
        bigmat[:, i-1] = stim_mat[:, i-nleads: i].flatten()
savefile = os.path.join(stimfolder, r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5-bigmat.pkl')

stim.bigmat_file = savefile
with open(os.path.join(stimfolder, stimfile), 'wb') as outfile:
    pickle.dump(stim, outfile, pickle.HIGHEST_PROTOCOL)

# --------------------------------------------get stimulus mtf --------------------------------------------------------
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim = pickle.load(f)
stim_mtf = stim.mtf()
savefile = re.sub('6.pkl', '6-mtf.pkl', stimfile)
with open(savefile, 'wb') as outfile:
    pickle.dump(stim_mtf, outfile, pickle.HIGHEST_PROTOCOL)

# option2. get mtf file from .mat file
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
matfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_mtf.mat'
mtf = loadmat(os.path.join(stimfolder, matfile))
savefile = re.sub('mtf.mat', 'mtf.pkl', matfile)
with open(os.path.join(stimfolder, savefile), 'wb') as outfile:
    pickle.dump(mtf, outfile, pickle.HIGHEST_PROTOCOL)


