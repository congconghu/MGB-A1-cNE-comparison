import os
import glob
import pickle
import re
import mat73

import session_toolbox as mtp
import helper

# ------------------------------------------- pickle single units-------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-Matlab'
files = glob.glob(datafolder + r'\*split.mat', recursive=False)

for idx, file in enumerate(files):
    print('({}/{})Save pickle file:'.format(idx+1, len(files)), file)
    session = mtp.Session()
    session.read_mat_file(file)
    file_pkl = re.sub('-spk-curated-split.mat', '.pkl', file)
    file_pkl = re.sub('Matlab', 'pkl', file_pkl)
    session.save_json_file(file_pkl)

# load pickle file
with open(file_pkl, 'rb') as f:
    session = pickle.load(f)

# ------------------------------------------- pickle stimulus----------------------------------------------------------
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5_stim.mat'
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


