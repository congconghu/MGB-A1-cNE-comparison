import glob
import os
import pickle
import re
import ne_toolbox as netools
import session_toolbox as st
from session_toolbox import save_su_df, save_session_df

# ++++++++++++++++++++++++++++++++++++++++++++ single unit properties ++++++++++++++++++++++++++++++++++++++++++++++++
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)

stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
# get stimulus for strf calculation (spectrogram)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_strf = pickle.load(f)
stim_strf.down_sample(df=10)
# get stimulus for crh calculation (mtf)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_mtf.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_crh = pickle.load(f)

for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    print('({}/{}) processing {}'.format(idx + 1, len(files), file))
    print('get unit positions')
    session.get_unit_position()
    
    print('get 5ms binned strf')
    session.get_strf(stim_strf)
    print('get strf properties')
    session.get_strf_properties()
    print('get strf ptd')
    session.get_strf_ptd()
    print('get strf nonlinearity')
    session.get_strf_nonlinearity(stim_strf)
    print('get strf mutual information')
    session.get_strf_mi(stim_strf)
    print('get strf RI')
    session.get_strf_ri(stim_strf)
    print('get strf sig')
    session.get_strf_significance(criterion='z', thresh=3)
    
    print('get crh')
    session.get_crh(stim_crh)
    print('get crh properties')
    session.get_crh_properties()
    print('get crh moranI')
    session.get_crh_morani()
    print('get crh RI')
    session.get_crh_ri(stim_crh)
    print('get crh sig')
    session.get_crh_significance(criterion='z', thresh=3)
    
    session.save_pkl_file(session.file_path)

save_su_df()
save_session_df()


# ++++++++++++++++++++++++++++++++++++++++++++++ cNE analysis  ++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    print('({}/{}) get cNE activities and members for {}'.format(idx + 1, len(files), file))
    if not hasattr(ne, 'ne_activity'):
        print('get members and activity')
        ne.get_members()
        ne.get_activity(member_only=True)
        ne.get_activity_thresh()
        ne.save_pkl_file(ne.file_path)
    if not hasattr(ne, 'ne_units'):
        print('get ne spikes')
        ne.get_ne_spikes(alpha=alpha)
        ne.save_pkl_file(ne.file_path)
    
    ne.save_pkl_file(ne.file_path)
    
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

# ------------------------------------match split cNEs and get null distribution of corr------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*split.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne_split = pickle.load(f)
    print('({}/{})  processing ne on split blocks for {}'.format(idx + 1, len(files), file))
    
    # match split cNEs
    print('match ICweights of split cNEs')
    netools.get_split_ne_ic_weight_match(ne_split)
    netools.get_ic_weight_corr(ne_split)
    
    # get null distribution of ICweight correlations
    print('get null ICweights')
    netools.get_split_ne_null_ic_weight(ne_split, nshift=1000)
    netools.get_null_ic_weight_corr(ne_split)
    netools.get_ic_weight_corr_thresh(ne_split)
    
    with open(file, 'wb') as output:
        pickle.dump(ne_split, output, pickle.HIGHEST_PROTOCOL)

# --------------------------subsample to get same numbe of neurons in each recording----------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
netools.sub_sample_split_ne(files, datafolder)

# -------------------------------save split cNE parameters to Pandas DataFrame ----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
savefolder = r'E:\Congcong\Documents\data\comparison\data-summary'
files = glob.glob(datafolder + r'\*split.pkl', recursive=False)
netools.get_split_ne_df(files, savefolder)
files = glob.glob(datafolder + r'\*sub10.pkl', recursive=False)
netools.get_split_ne_null_df(files, savefolder)


# ++++++++++++++++++++++++++++++++++++++++++++++ cNE stim response ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------------------------- get cNE stimulus responses ---------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim = pickle.load(f)
stim.down_sample(df=10)

# get stimulus for crh calculation (mtf)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_mtf.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_crh = pickle.load(f)
    
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    print('({}/{}) get cNE response properties for {}'.format(idx + 1, len(files), file))
    
    # get ne strf and strf properties
    print('get 5ms binned strf')
    ne.get_strf(stim)
    ne.get_strf_ri(stim)
    ne.get_strf_properties()
    ne.get_strf_significance(criterion='z', thresh=3)
    ne.get_strf_nonlinearity(stim)
    ne.get_strf_info(stim)
    
    # get ne crh and crh properties
    print('get crh')
    ne.get_crh(stim_crh)
    ne.get_crh_ri(stim_crh)
    ne.get_crh_properties()
    ne.get_crh_significance(criterion='z', thresh=3)
    
    ne.save_pkl_file(ne.file_path)
    
# ------------------------------------------reliability of stimulus response-------------------------------------------
savefolder = r'E:\Congcong\Documents\data\comparison\data-summary'
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
# get stimulus for strf calculation (spectrogram)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_strf = pickle.load(f)
stim_strf.down_sample(df=10)
# get stimulus for crh calculation (mtf)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_mtf.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_crh = pickle.load(f)

st.ri_ne_neuron_subsample(stim_strf, stim_crh, datafolder, savefolder)

# ------------------------------------------------strf ptd-------------------------------------------------------------
