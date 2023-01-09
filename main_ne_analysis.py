import glob
import os
import pickle
import re
import ne_toolbox as netools


# ++++++++++++++++++++++++++++++++++++++++++++ single unit properties ++++++++++++++++++++++++++++++++++++++++++++++++
# get unit positions
datafolder = '/Users/hucongcong/Documents/UCSF/data/data-pkl'
files = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
for file in files:
    with open(file, 'rb') as f:
        session = pickle.load(f)
    session.get_unit_position()
    with open(file, 'wb') as f:
        pickle.dump(session, f)

# ++++++++++++++++++++++++++++++++++++++++++++++ cNE properties  ++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

# ---------------------------------------- get xcorr of member and nonmember pairs -------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
xcorr = netools.get_member_nonmember_xcorr(files, df=2, maxlag=200)
xcorr.to_json(r'E:\Congcong\Documents\data\comparison\data-summary\member_nonmember_pair_xcorr.json')


# ++++++++++++++++++++++++++++++++++++ cNE spon/stim stability analysis  +++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------------------------get ne of split activities -------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    print('({}/{}) get cNEs of split activities for {}'.format(idx + 1, len(files), file))
    ne_split = session.get_ne_split(df=20)
    if 'dmr1' in ne_split:
        # get split cNE members
        for key, ne in ne_split.items():
            if key == 'dmr_first':
                continue
            ne.get_members()
        savefile_path = re.sub(r'fs20000.pkl', r'fs20000-ne-20dft-split.pkl', session.file_path)
        with open(savefile_path, 'wb') as output:
            pickle.dump(ne_split, output, pickle.HIGHEST_PROTOCOL)

# -----------------------------------------match split cNEs------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*split.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne_split = pickle.load(f)
    print('({}/{}) match ICweights of split cNEs for {}'.format(idx + 1, len(files), file))
    ne_split = netools.get_split_ne_ic_weight_match(ne_split)
    netools.get_ic_weight_corr(ne_split)
    with open(file, 'wb') as output:
        pickle.dump(ne_split, output, pickle.HIGHEST_PROTOCOL)

# -------------------------------get null distribution of ICweight correlations ----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*split.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne_split = pickle.load(f)
    print('({}/{}) get null ICweights of split cNEs for {}'.format(idx + 1, len(files), file))
    netools.get_split_ne_null_ic_weight(ne_split, nshift=1000)
    netools.get_null_ic_weight_corr(ne_split)
    netools.get_ic_weight_corr_thresh(ne_split)
    with open(file, 'wb') as output:
        pickle.dump(ne_split, output, pickle.HIGHEST_PROTOCOL)

# save split cNE information in pandas DataFrame
