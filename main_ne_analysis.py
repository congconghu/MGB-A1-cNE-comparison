import glob
import os
import pickle
import re
import ne_toolbox as netools
import session_toolbox as st
from session_toolbox import save_su_df, save_session_df
import numpy as np
import pandas as pd

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

# -------------------------------get frequency span of cNEs on each block ----------------------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/summary'
savefolder = r'/Users/hucongcong/Documents/UCSF/data/summary'
netools.get_split_ne_freq_span(datafolder, savefolder)


# ++++++++++++++++++++++++++++++++++++ cNE significanc: data shuffling  +++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------ get cNEs and number of cNEs from shuffled data ---------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    # cNE analysis
    print('({}/{}) Get shuffled cNEs for {}'.format(idx+1, len(files), file))
    for stim in ('dmr', 'spon'):
        savefile_path = re.sub(r'fs20000.pkl', 'fs20000-ne-20dft-{}-shuffled.pkl'.format(stim), session.file_path)
        ne = []
        n_ne = np.zeros(10)
        for idx in range(10):
            ne.append(session.get_ne(df=20, stim=stim, shuffle=True))
            if ne[-1] is not None:
                ne[-1].get_members()
                n_ne[idx] = len(ne[-1].ne_members)
        with open(savefile_path, 'wb') as f:
            pickle.dump({'ne': ne, 'n_ne': n_ne}, f)
netools.get_num_cne_vs_shuffle()

# --------------------------------------- stability across dmr/spon on shuffled data --------------------------------
# real data
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*ne-20dft-dmr.pkl', recursive=False)
nfile_MGB = 0
nfile_A1 = 0
for file in files:
   
    file_spon = re.sub('20dft-dmr', '20dft-spon', file)
    if not os.path.exists(file_spon):
        continue
    
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    with open(file_spon, 'rb') as f:
        ne_spon = pickle.load(f)
    corrmat, order, _ = netools.match_ic_weight(ne.patterns, ne_spon.patterns)
    ne.corrmat = corrmat
    ne.pattern_order = order
    ne.save_pkl_file()
    
# shuffled data
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*ne-20dft-dmr.pkl', recursive=False)
for idx, file in enumerate(files):
    file_spon = re.sub('20dft-dmr', '20dft-spon', file)
    if not os.path.exists(file_spon):
        continue
    
    with open(file, 'rb') as f:
        ne_dmr = pickle.load(f)
    with open(file_spon, 'rb') as f:
        ne_spon = pickle.load(f)

    print('({}/{})  processing ne on split blocks for {}'.format(idx + 1, len(files), file))
    
    # get null distribution of ICweight correlations
    print('get null ICweights')
    for ne in [ne_dmr, ne_spon]:
        ne.get_sham_patterns(nshift=1000)
    
    with open(file, 'wb') as output:
        pickle.dump(ne_dmr, output)
    with open(file_spon, 'wb') as output:
        pickle.dump(ne_spon, output)

# get correlation threshold
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*ne-20dft-dmr.pkl', recursive=False)
for idx, file in enumerate(files):
    file_spon = re.sub('20dft-dmr', '20dft-spon', file)
    if not os.path.exists(file_spon):
        continue
    
    with open(file, 'rb') as f:
        ne_dmr = pickle.load(f)
    with open(file_spon, 'rb') as f:
        ne_spon = pickle.load(f)

    print('({}/{})  processing ne on split blocks for {}'.format(idx + 1, len(files), file))
    
    # get null distribution of ICweight correlations
    n_ne = ne_dmr.patterns_sham.shape[0]
    corr = np.abs(np.corrcoef(
        x=ne_dmr.patterns_sham,
        y=ne_spon.patterns_sham))[:n_ne, n_ne:]
    ne_dmr.corr_null = corr.flatten()
    ne_dmr.corr_thresh = np.percentile(abs(ne_dmr.corr_null), 99)
    
    with open(file, 'wb') as output:
        pickle.dump(ne_dmr, output)


# --------------------- save significance of pattern correlation for shuffled data -------------------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/summary'
ne_real = pd.read_json(os.path.join(datafolder, 'cNE_matched.json'))
ne_shuffled = pd.read_json(os.path.join(datafolder, 'cNE_matched_shuffled.json'))
corr_thresh = ne_real[['exp', 'probe', 'depth', 'corr_thresh']]
corr_thresh = corr_thresh.drop_duplicates()
ne_shuffled = pd.merge(ne_shuffled, corr_thresh,  how='left', on=['exp', 'probe', 'depth'])
ne_shuffled['corr_sig'] = ne_shuffled['pattern_corr'] > ne_shuffled['corr_thresh']
ne_shuffled.to_json(os.path.join(datafolder, 'cNE_matched_shuffled.json'))


# ++++++++++++++++++++++++++++++++++++ cNE significanc: data shuffling for ne stability  +++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------  ---------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*split.pkl', recursive=False)
for idx, file in enumerate(files):
    print('({}/{}) Get shuffled cNEs for {}'.format(idx+1, len(files), file))
    
    # cNEs from shuffled data
    ne_split = netools.get_split_shuffled_ne(file)
    # get "stability" false positive cNEs
    ne_split = netools.get_ic_weight_corr_shuffled(ne_split)
    
    savefile = re.sub('split', 'split-shuffled', file)
    with open(savefile, 'wb') as f:
        pickle.dump(ne_split, f)
        
# get number of cNEs from shuffled data
netools.get_num_cne_vs_shuffle_split()
# save icweight correlation of real and shuffled data in dataframe
netools.get_ne_split_real_vs_shuffle_corr()


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
    ne.get_strf_ptd()
    
    # get ne crh and crh properties
    print('get crh')
    ne.get_crh(stim_crh)
    ne.get_crh_ri(stim_crh)
    ne.get_crh_properties()
    ne.get_crh_significance(criterion='z', thresh=3)
    ne.get_crh_morani()
    
    ne.save_pkl_file(ne.file_path)
    
# --------------------------------------get stimulus response with subsampled spike trains--------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
savefolder = r'E:\Congcong\Documents\data\comparison\data-summary\subsample'

stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
# get stimulus for strf calculation (spectrogram)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_strf = pickle.load(f)
stim_strf.down_sample(df=10)
# get stimulus for crh calculation (mtf)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5-mtf.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_crh = pickle.load(f)

st.ne_neuron_subsample(stim_strf, stim_crh, datafolder, savefolder)

# ------------------------------- get strf of random groups of neurons / coincident spikes -------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
savefolder = r'E:\Congcong\Documents\data\comparison\data-summary'

stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
# get stimulus for strf calculation (spectrogram)
stimfile = r'rn1-500flo-40000fhi-0-4SM-0-40TM-40db-96khz-48DF-15min-seed190506_DFt1_DFf5.pkl'
with open(os.path.join(stimfolder, stimfile), 'rb') as f:
    stim_strf = pickle.load(f)
stim_strf.down_sample(df=10)
for control in ('random_group', 'coincident_spike'):
    st.ne_neuron_strf_control(stim_strf, control, savefolder=os.path.join(savefolder, control), nrepeat=1000)
    st.ne_neuron_strf_control_combine(control, datafolder=savefolder)


# ++++++++++++++++++++++++++++++++++++++++++++++ cNE stability across binsize ++++++++++++++++++++++++++++++++++++++++++
# -------------------------------------------- get cNE under different binsizes ----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
dfs = np.array([2, 5, 20, 40, 80, 160])*2
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    # cNE analysis
    print('({}/{}) Get cNEs for {}'.format(idx+1, len(files), file))
    for df in dfs:
        for stim in ('dmr', 'spon'):
            savefile_path = re.sub(r'fs20000.pkl', 'fs20000-ne-{}dft-{}.pkl'.format(df, stim), session.file_path)
            ne = session.get_ne(df=20, stim=stim)
            if ne:
                ne.save_pkl_file(savefile_path)

# ------------------------------------------- match cNE patterns in each recording -------------------------------------
# match cNE patterns to different bin sizes
stim = 'spon'
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl\binsize\{}'.format(stim)
savefolder = r'E:\Congcong\Documents\data\comparison\data-pkl\binsize\ic_match_tbins'
dfs = np.array([2, 5, 10, 20, 40, 80, 160])*2
for df in dfs:
    files = glob.glob(datafolder + r'/*-{}dft-{}.pkl'.format(df, stim), recursive=False)
    for idx, file in enumerate(files):
        savefile = re.sub(f'{stim}.pkl', f'{stim}-ic_match_tbins.pkl',os.path.basename(file))
        print(r'{}/{} Matching time bins for {}'.format(idx+1, len(files), savefile))
        ic_matched = netools.ICweight_match_binsize(datafolder, file, dfs)
        with open(os.path.join(savefolder, savefile), 'wb') as f:
            pickle.dump(ic_matched, f)

# -------------------------------------------- data summary saved to dataframe -----------------------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/data-pkl-binsize'
savefolder = r'/Users/hucongcong/Documents/UCSF/data/summary'
dfs = np.array([2, 5, 20, 40, 80, 160])*2
netools.batch_save_icweight_binsize_corr_to_dataframe(datafolder, savefolder, dfs)

# ------------------------------------get ccg of members identified under diffrent bin sizes ---------------------------
datafolder = r'/Users/hucongcong/Documents/UCSF/data/data-pkl-binsize'
jsonfile = r'/Users/hucongcong/Documents/UCSF/data/summary/icweight_corr_binsize.json'
df_ref = 20
netools.batch_save_icweight_ccg_binsize(datafolder, jsonfile, savefolder, df_ref)

# ++++++++++++++++++++++++++++++++++++++++++++++ cNE and UP/DOWN states ++++++++++++++++++++++++++++++++++++++++++
# ---------------------------------------------get up.down spikes ------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl\up_down'
files = glob.glob(datafolder + r'\*-20dft-dmr.pkl', recursive=False)
for idx, file in enumerate(files):
    
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    
    ne.get_up_down_spikes(datafolder)
    filename = re.findall('\d{6}_\d{6}.*', ne.file_path)[0]
    ne.save_pkl_file(os.path.join(datafolder, filename))
    
    session = ne.get_session_data()
    session.get_up_down_spikes(datafolder)
    filename = re.findall('\d{6}_\d{6}.*',  session.file_path)[0]
    session.save_pkl_file(os.path.join(datafolder, filename))

# get up spktrain 
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl\up_down_spon'
files = glob.glob(datafolder + r'\*20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    # save spktrains
    print('({}/{}) Save spktrain_up for {}'.format(idx+1, len(files), file))
    session.save_spktrain_up_state()


# get cNE from spktrain_up
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl\up_down_spon'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)

    # cNE analysis
    print('({}/{}) Get cNEs for {}'.format(idx+1, len(files), file))
    stim = 'spon_up'
    savefile_path = re.sub(r'fs20000.pkl', 'fs20000-ne-20dft-{}.pkl'.format(stim), session.file_path)
    ne = session.get_ne(df=20, stim=stim)
    if ne:
        ne.get_members()
        ne.save_pkl_file(savefile_path)

# match patterns of cNEs
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl\up_down'
files = glob.glob(datafolder + r'\*ne-20dft-dmr_up.pkl', recursive=False)
nfile_MGB = 0
nfile_A1 = 0
for file in files:
   
    file_all = re.sub('20dft-dmr_up', '20dft-dmr', file)
    if not os.path.exists(file_all):
        continue
    
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    with open(file_all, 'rb') as f:
        ne_all = pickle.load(f)
    corrmat, order, _ = netools.match_ic_weight(ne.patterns, ne_all.patterns)
    ne.corrmat = corrmat
    ne.pattern_order = order
    ne.save_pkl_file()
    
st.save_matched_ne_df(datafolder=datafolder, key='dmr_up')
