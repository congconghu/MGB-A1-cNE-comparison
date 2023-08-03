# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:50:53 2023

@author: Congcong
"""
import glob
import os
import pickle

datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)

stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
# get stimulus for strf calculation (spectrogram)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_strf = pickle.load(f)
stim_strf.down_sample(df=10)

for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    print('({}/{}) processing {}'.format(idx + 1, len(files), file))

    print('get strf mutual information')
    session.get_strf_mi(stim_strf)
    
    session.save_pkl_file(session.file_path)