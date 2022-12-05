import data_from_mat_to_pkl as mtp
import helper
import os
import pickle
import re

# ------------------------------------------- pickle stimulus----------------------------------------------------------
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_stim.mat'
stimfile_pkl = re.sub('_stim.mat', '.pkl', stimfile)

stimfile_path = os.path.join(stimfolder, stimfile)
stimfile_pkl_path = os.path.join(stimfolder, stimfile_pkl)

# save stimulus files from mat to pickle
# takes ~20s to lead mat file
# takes ~4s to lead pickle file
stim = mtp.Stimulus()
with helper.Timer():
    stim.read_mat_file(stimfile_path)
stim.save_pkl_file(stimfile_pkl_path)

# load stimulus files from pickle file
with helper.Timer():
    with open(stimfile_pkl_path, 'rb') as f:
        stim = pickle.load(f)

# ------------------------------------------- pickle single units-------------------------------------------------------
