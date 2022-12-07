import glob
import os
import pickle
import re

# ------------------------------------------------ get cNEs-------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
nefolder = datafolder

# load stimulus
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim = pickle.load(f)

files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    # save spktrains
    if not hasattr(session, 'spktrain_dmr'):
        print('({}/{}) Save spktrain for {}'.format(idx, len(files), file))
        session.save_spktrain_from_stim(stim.stim_mat.shape[1])
    # cNE analysis
    savefile_path = re.sub(r'fs20000.pkl', r'fs20000-ne-20dft.pkl', session.file_path)
    if not os.path.exists(savefile_path):
        ne = session.get_ne(df=20)
        ne.save_pkl_file(savefile_path)

# ----------------------------------------- get cNE members and activities ---------------------------------------------
nefolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*20dft.pkl', recursive=False)
for idx, file in enumerate(files):
    print('({}/{}) get members and activities for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)



