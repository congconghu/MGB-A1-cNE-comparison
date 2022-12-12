import glob
import os
import pickle
import re


# ------------------------------------------------ get cNEs-------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'

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
        print('({}/{}) Save spktrain for {}'.format(idx+1, len(files), file))
        session.save_spktrain_from_stim(stim.stim_mat.shape[1])
    # cNE analysis
    print('({}/{}) Get cNEs for {}'.format(idx+1, len(files), file))
    for stim in ('dmr', 'spon'):
        savefile_path = re.sub(r'fs20000.pkl', 'fs20000-ne-20dft-{}.pkl'.format(stim), session.file_path)
        ne = session.get_ne(df=20, stim=stim)
        if ne:
            ne.save_pkl_file(savefile_path)

# ----------------------------------------- get cNE members and activities ---------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

alpha = 99.5
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim = pickle.load(f)
stim.down_sample(df=10)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    if not hasattr(ne, 'ne_activity'):
        print('({}/{}) get members and activity for {}'.format(idx + 1, len(files), file))
        ne.get_members()
        ne.get_activity(member_only=True)
        ne.get_activity_thresh()
        ne.save_pkl_file(ne.file_path)
    if not hasattr(ne, 'ne_units'):
        print('({}/{}) get ne spikes for {}'.format(idx + 1, len(files), file))
        ne.get_ne_spikes(alpha=alpha)
        ne.save_pkl_file(ne.file_path)

    print('({}/{}) get 5ms binned strf for {}'.format(idx + 1, len(files), file))
    ne.get_strf(stim)
    ne.save_pkl_file(ne.file_path)

# ------------------------------------------------ get 5ms strf for each session----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)

stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim = pickle.load(f)
stim.down_sample(df=10)

for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    print('({}/{}) get 5ms binned strf for {}'.format(idx + 1, len(files), file))
    session.get_strf(stim)
    session.save_pkl_file(session.file_path)
