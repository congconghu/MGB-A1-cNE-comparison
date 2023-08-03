import itertools
import re
import glob
import os
import pickle
import random as rand
from itertools import combinations
from copy import deepcopy

import scipy
import numpy as np
import pandas as pd

from scipy.ndimage import convolve
from scipy.stats import zscore
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


def detect_cell_assemblies(spktrain):
    spktrain_z = zscore(spktrain, 1)
    pca = PCA()
    pca.fit(np.transpose(spktrain_z))

    # get threshold for significant PCs (based on Marcenko-Patur distribution)
    thresh = get_pc_thresh(spktrain)
    num_ne = sum(pca.explained_variance_ > thresh)
    if num_ne > 0:
        patterns = fast_ica(spktrain_z, num_ne)
        return normalize_pattern(patterns)
    else:
        return None


@ignore_warnings(category=ConvergenceWarning)
def fast_ica(spktrain_z, num_ne, niter=500):
    """get cNE patterns using ICA"""
    # step1: PCA
    pca = PCA()
    pca.fit(np.transpose(spktrain_z))
    # get subspace spanned by num_ne PCa
    eigen_vectors = pca.components_[:num_ne]
    eigen_values = pca.explained_variance_[:num_ne]
    whiten_mat = np.diag(1 / np.sqrt(eigen_values)) @ eigen_vectors
    dewhiten_mat = np.transpose(eigen_vectors) @ np.diag(np.sqrt(eigen_values))
    spktrain_w = whiten_mat @ spktrain_z  # project spktrain_z to first n PCs and scale variance to 1
    # step2: ICA
    ica = FastICA(tol=1e-10, max_iter=niter, random_state=0, whiten=False)
    ica.fit(np.transpose(spktrain_w))
    unmixing = ica._unmixing
    return unmixing @ np.transpose(dewhiten_mat)


def normalize_pattern(patterns):
    """Normalize length of the patterns to be 1 and make the highest absolute deviation to be positive"""
    patterns = [x / np.sqrt(np.sum(x ** 2)) for x in patterns]
    patterns = [-x if x[abs(x).argmax()] < 0 else x for x in patterns]
    return np.array(patterns)


def get_activity(spktrain, pattern, member_only=False, members=[]):
    # when only considering member neurons, set nonmember weights to 0
    if member_only:
        pattern = np.array([pattern[x] for x in members])
        spktrain = np.array([spktrain[x] for x in members])

    spktrain_z = zscore(spktrain, 1)
    projector = np.outer(pattern, pattern)
    np.fill_diagonal(projector, 0)
    activity = np.zeros(spktrain.shape[1])
    for i in range(spktrain.shape[1]):
        if np.sum(spktrain[:, i]) > 1:
            spike_pattern = spktrain_z[:, i]
            activity[i] = spike_pattern @ projector @ np.transpose(spike_pattern)
    return activity


def get_ne_spikes(activity, thresh, spiketimes, edges):
    idx_event = np.where(activity > thresh)[0]  # get index of time bins when activities cross threshold
    n_member = len(spiketimes)
    for i in range(n_member):
        spiketimes[i] = get_binned_spiketimes(spiketimes[i], edges)

    ne_spikes = np.array([])
    ne_spikes_member = [np.array([]) for i in range(n_member)]
    for idx in idx_event:
        ne_spikes_tmp = [spiketimes[i][idx] for i in range(n_member)]
        for i in range(n_member):
            ne_spikes_member[i] = np.append(ne_spikes_member[i], ne_spikes_tmp[i])
        ne_spikes_tmp = np.concatenate(ne_spikes_tmp)
        ne_spikes = np.append(ne_spikes, ne_spikes_tmp.max())
    return ne_spikes, ne_spikes_member


def get_binned_spiketimes(spiketimes, edges):
    spiketimes_binned = []
    if not type(edges) == list:
        edges = [edges]
    for edge in edges:
        edge = edge.flatten()
        for i in range(len(edge) - 1):
            spiketimes_binned.append(spiketimes[(spiketimes >= edge[i]) & (spiketimes < edge[i + 1])])
    return spiketimes_binned


def get_pc_thresh(spktrain):
    q = spktrain.shape[1] / spktrain.shape[0]
    thresh = (1 + np.sqrt(1 / q)) ** 2
    return thresh


def get_member_nonmember_xcorr(files, df=2, maxlag=200):
    xcorr = {'xcorr': [], 'corr': [], 'member': [], 'stim': [], 'region': [], 'exp': []}
    for idx, file in enumerate(files):
        print('({}/{}) get member and nonmmebers xcorr for {}'.format(idx + 1, len(files), file))

        with open(file, 'rb') as f:
            session = pickle.load(f)

        n_neuron = len(session.units)
        all_pairs = set(combinations(range(n_neuron), 2))

        ne_file_path = re.sub('fs20000', 'fs20000-ne-20dft*', session.file_path)
        nefiles = glob.glob(ne_file_path)
        for nefile in nefiles:

            # load ne data
            with open(nefile, 'rb') as f:
                ne = pickle.load(f)

            # get region of the recording
            if session.depth > 2000:
                region = 'MGB'
            else:
                region = 'A1'

            # get stimulus condition of the cNEs
            if nefile.endswith('dmr.pkl'):
                stim = 'dmr'
            elif nefile.endswith('spon.pkl'):
                stim = 'spon'

            member_pairs = get_member_pairs(ne)
            nonmember_pairs = all_pairs.difference(member_pairs)

            spktrain, _ = session.downsample_spktrain(df=df, stim=stim)
            spktrain_shift = np.roll(spktrain, -maxlag, axis=1)
            spktrain_shift = spktrain_shift[:, :-2 * maxlag]
            corr_mat = np.corrcoef(ne.spktrain)
            for u1, u2 in member_pairs:
                c = np.correlate(spktrain[u1], spktrain_shift[u2], mode='valid')
                xcorr['xcorr'].append(np.array(c).astype('int16'))
                xcorr['corr'].append(corr_mat[u1][u2])
                xcorr['member'].append(True)
                xcorr['stim'].append(stim)
                xcorr['region'].append(region)
                xcorr['exp'].append(session.exp)
            for u1, u2 in nonmember_pairs:
                c = np.correlate(spktrain[u1], spktrain_shift[u2], mode='valid')
                xcorr['xcorr'].append(c)
                xcorr['corr'].append(corr_mat[u1][u2])
                xcorr['member'].append(False)
                xcorr['stim'].append(stim)
                xcorr['region'].append(region)
                xcorr['exp'].append(session.exp)

    xcorr = pd.DataFrame(xcorr)
    return xcorr


def get_member_pairs(ne):
    member_pairs = set()
    for members in ne.ne_members.values():
        member_pairs.update(set(combinations(members, 2)))
    return member_pairs


# ------------ get cNE on split activities and related analysis of dmr/spon stability ------------------------
def get_split_ne_ic_weight_match(ne_split):
    stims = ('dmr', 'spon')
    patterns = {'dmr': [], 'spon': []}
    for stim in stims:
        for idx in range(2):
            patterns[stim].append(ne_split[stim + str(idx)].patterns)

    # reorder NEs based on within condition correlation value
    orders = {'dmr': [], 'spon': []}
    for stim in stims:
        _, orders[stim], patterns[stim] = match_ic_weight(*patterns[stim])

    # get correlation matrix with both conditions
    patterns_all = []
    for i in range(2):
        patterns_all.append(np.concatenate([patterns['dmr'][i], patterns['spon'][i]]))
    n_ne = patterns_all[0].shape[0]
    corr_mat = np.abs(np.corrcoef(*patterns_all))[:n_ne, n_ne:]
    corr_mat, order = corr_mat_reorder_cross(corr_mat, ne_split['dmr_first'], orders)
    ne_split['corr_mat'] = corr_mat
    ne_split['order'] = order


def match_ic_weight(patterns1, patterns2):
    n1, n2 = patterns1.shape[0], patterns2.shape[0]
    n_ne = min([n1, n2])
    corr_mat = np.abs(np.corrcoef(patterns1, patterns2))[:n1, n1:]
    order1, order2 = corr_mat_reorder(corr_mat)
    patterns1 = patterns1[np.array(order1)]
    patterns2 = patterns2[np.array(order2)]
    corr_mat = np.abs(np.corrcoef(patterns1, patterns2))[:n_ne, n_ne:]
    return corr_mat, [order1, order2], [patterns1, patterns2]


def corr_mat_reorder(corr_mat):
    order1, order2 = [], []
    n_ne = min(corr_mat.shape)
    for _ in range(n_ne):
        row, col = np.unravel_index(corr_mat.argmax(), corr_mat.shape)
        corr_mat[row, :] = 0
        corr_mat[:, col] = 0
        order1.append(row)
        order2.append(col)
    return order1, order2


def corr_mat_reorder_cross(corr_mat, dmr_first, orders):
    n_ne_dmr = len(orders['dmr'][0])
    if dmr_first:
        corr_dmr_spon = corr_mat[n_ne_dmr:, :n_ne_dmr]
        row_add, col_add = n_ne_dmr, 0
    else:
        corr_dmr_spon = corr_mat[:n_ne_dmr, n_ne_dmr:]
        row_add, col_add = 0, n_ne_dmr

    n_ne_cross = min(corr_dmr_spon.shape)
    for i in range(n_ne_cross):
        row, col = np.unravel_index(corr_dmr_spon.argmax(), corr_dmr_spon.shape)
        if row != i:
            corr_mat, orders = swap_lines(corr_mat, row_add + row, row_add + i, orders)
        if col != i:
            corr_mat, orders = swap_lines(corr_mat, col_add + col, col_add + i, orders)
        if dmr_first:
            corr_dmr_spon = np.copy(corr_mat[n_ne_dmr:, :n_ne_dmr])
        else:
            corr_dmr_spon = np.copy(corr_mat[:n_ne_dmr, n_ne_dmr:])
        corr_dmr_spon[:i + 1, :] = 0
        corr_dmr_spon[:, :i + 1] = 0
    return corr_mat, orders


def swap_lines(corr_mat, idx1, idx2, orders):
    n_dmr = len(orders['dmr'][0])
    # swap row
    corr_mat[[idx1, idx2]] = corr_mat[[idx2, idx1]]
    order_combined = orders['dmr'][0] + orders['spon'][0]
    order_combined[idx1], order_combined[idx2] = order_combined[idx2], order_combined[idx1]
    orders['dmr'][0] = order_combined[:n_dmr]
    orders['spon'][0] = order_combined[n_dmr:]

    corr_mat[:, [idx1, idx2]] = corr_mat[:, [idx2, idx1]]
    order_combined = orders['dmr'][1] + orders['spon'][1]
    order_combined[idx1], order_combined[idx2] = order_combined[idx2], order_combined[idx1]
    orders['dmr'][1] = order_combined[:n_dmr]
    orders['spon'][1] = order_combined[n_dmr:]
    return corr_mat, orders


def get_split_ne_null_ic_weight(ne_split, nshift=1000):
    stims = ['dmr0', 'dmr1', 'spon0', 'spon1']
    for stim in stims:
        ne = ne_split[stim]
        ne.get_sham_patterns(nshift=nshift)


def get_null_ic_weight_corr(ne_split):
    corr_null = {'dmr': [], 'spon': [], 'cross': []}
    for stim in ('spon', 'dmr'):
        n_ne = ne_split[stim + '0'].patterns_sham.shape[0]
        corr = np.abs(np.corrcoef(
            x=ne_split[stim + '0'].patterns_sham,
            y=ne_split[stim + '1'].patterns_sham))[:n_ne, n_ne:]
        corr_null[stim] = corr.flatten()
    if ne_split['dmr_first']:
        n_ne = ne_split['spon0'].patterns_sham.shape[0]
        corr = np.abs(np.corrcoef(
            x=ne_split['spon0'].patterns_sham,
            y=ne_split['dmr1'].patterns_sham))[:n_ne, n_ne:]
    else:
        n_ne = ne_split['spon1'].patterns_sham.shape[0]
        corr = np.abs(np.corrcoef(
            x=ne_split['spon1'].patterns_sham,
            y=ne_split['dmr0'].patterns_sham))[:n_ne, n_ne:]
    corr_null['cross'] = corr.flatten()
    ne_split['corr_null'] = corr_null


def get_ic_weight_corr_thresh(ne_split, alpha=99):
    corr_thresh = {'dmr': [], 'spon': [], 'cross': []}
    for stim, corr in ne_split['corr_null'].items():
        corr_thresh[stim] = np.percentile(corr, alpha)
    ne_split['corr_thresh'] = corr_thresh


def get_ic_weight_corr(ne_split):
    corr = {'dmr': [], 'spon': [], 'cross': []}
    n_ne = min(ne_split['dmr0'].patterns.shape[0], ne_split['dmr1'].patterns.shape[0])
    corr['dmr'] = ne_split['corr_mat'][:n_ne, :n_ne].diagonal()
    corr['spon'] = ne_split['corr_mat'][n_ne:, n_ne:].diagonal()
    if ne_split['dmr_first']:
        corr['cross'] = ne_split['corr_mat'][n_ne:, :n_ne].diagonal()
    else:
        corr['cross'] = ne_split['corr_mat'][:n_ne, n_ne:].diagonal()
    ne_split['corr'] = corr


def get_split_ne_df(files, savefolder):
    ne = pd.DataFrame(columns=['exp', 'probe', 'region', 'stim', 'dmr_first', 'idx1', 'idx2',
                               'pattern1', 'pattern2', 'corr', 'corr_thresh'])

    for idx, file in enumerate(files):
        print('({}/{}) get split cNE patterns for {}'.format(idx + 1, len(files), file))
        exp = re.findall('\d{6}_\d{6}', file)[0]
        probe = re.findall('H\d{2}x\d{2}', file)[0]
        region = 'A1' if probe == 'H22x32' else 'MGB'
        with open(file, 'rb') as f:
            ne_split = pickle.load(f)

        new_rows = get_matching_patterns_for_df(ne_split, exp, probe, region)
        ne = pd.concat([ne, new_rows], ignore_index=True)

    ne.to_json(os.path.join(savefolder, 'split_cNE.json'))


def get_matching_patterns_for_df(ne_split, exp, probe, region):
    ne = pd.DataFrame(columns=['exp', 'probe', 'region', 'stim', 'dmr_first', 'idx1', 'idx2',
                               'pattern1', 'pattern2', 'corr', 'corr_thresh'])
    # get matching cNE patterns within stim conditions
    n_ne_dmr = len(ne_split['order']['dmr'][0])
    for stim in ('dmr', 'spon'):
        n_ne = len(ne_split['order'][stim][0])
        for i in range(n_ne):
            idx1 = ne_split['order'][stim][0][i]
            idx2 = ne_split['order'][stim][1][i]
            if stim == 'dmr':
                corr = ne_split['corr_mat'][i, i]
            elif stim == 'spon':
                corr = ne_split['corr_mat'][i + n_ne_dmr, i + n_ne_dmr]
            new_row = pd.DataFrame({'exp': exp, 'probe': probe, 'region': region, 'stim': stim,
                                    'dmr_first': ne_split['dmr_first'], 'idx1': idx1, 'idx2': idx2,
                                    'pattern1': [ne_split[stim + '0'].patterns[idx1]],
                                    'pattern2': [ne_split[stim + '1'].patterns[idx2]],
                                    'corr': corr, 'corr_thresh': ne_split['corr_thresh'][stim]})
            ne = pd.concat([ne, new_row], ignore_index=True)

    # get matching cNE patterns cross stimulus conditions
    n_ne = min([n_ne_dmr, len(ne_split['order']['spon'][0])])
    for i in range(n_ne):
        if ne_split['dmr_first']:
            # dmr1
            idx1 = ne_split['order']['dmr'][1][i]
            pattern1 = [ne_split['dmr1'].patterns[idx1]]
            # spon0
            idx2 = ne_split['order']['spon'][0][i]
            pattern2 = [ne_split['spon0'].patterns[idx2]]
            corr = ne_split['corr_mat'][n_ne_dmr + i, i]
        else:
            # spon1
            idx1 = ne_split['order']['spon'][1][i]
            pattern1 = [ne_split['spon1'].patterns[idx1]]
            # dmr0
            idx2 = ne_split['order']['dmr'][0][i]
            pattern2 = [ne_split['dmr0'].patterns[idx2]]
            corr = ne_split['corr_mat'][i, n_ne_dmr + i]

        new_row = pd.DataFrame({'exp': exp, 'probe': probe, 'region': region, 'stim': 'cross',
                                'dmr_first': ne_split['dmr_first'], 'idx1': idx1, 'idx2': idx2,
                                'pattern1': pattern1, 'pattern2': pattern2,
                                'corr': corr,
                                'corr_thresh': ne_split['corr_thresh']['cross']})
        ne = pd.concat([ne, new_row], ignore_index=True)
    return ne


def get_split_ne_freq_span(datafolder, savefolder):
    ne_split = pd.read_json(os.path.join(datafolder, 'split_cNE.json'))
    su = pd.read_json(os.path.join(datafolder, 'single_units.json'))
    su = su[su.strf_sig]
    for i in range(1, 3):
        freq_span = []
        for j in range(len(ne_split)):
            ne = ne_split.iloc[j]
            su_tmp = su[(su.exp == ne.exp) & (su.probe == ne.probe)]
            members = eval(f'ne.member{i}')
            freqs = []
            for member in members:
                try:
                    freqs.append(su_tmp[su_tmp['index'] == member].bf.item())
                except:
                    continue
            try:
                freq_span.append(np.log2(np.max(freqs) / np.min(freqs)))
            except:
                freq_span.append(None)
        ne_split['freq_span' + str(i)] = freq_span
    ne_split.to_json(os.path.join(savefolder, 'split_cNE.json'))
def sub_sample_split_ne(files, savefolder, n_neuron=10, n_sample=10):
    rand.seed(0)
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            session = pickle.load(f)

        if len(session.units) <= n_neuron:
            continue
        elif not hasattr(session, 'spktrain_spon'):
            continue

        print('({}/{}) get split cNEs {}'.format(i + 1, len(files), file))
        spktrain_dmr = deepcopy(session.spktrain_dmr)
        spktrain_spon = deepcopy(session.spktrain_spon)

        ne_split_subsample = []
        for sample in range(n_sample):
            print('{}/{}'.format(sample + 1, n_sample))
            new_idx = rand.sample(range(len(session.units)), n_neuron)
            session.spktrain_dmr = spktrain_dmr[new_idx]
            session.spktrain_spon = spktrain_spon[new_idx]
            ne_split = session.get_ne_split(df=20)
            get_split_ne_ic_weight_match(ne_split)
            get_ic_weight_corr(ne_split)
            get_split_ne_null_ic_weight(ne_split, nshift=1000)
            get_null_ic_weight_corr(ne_split)
            get_ic_weight_corr_thresh(ne_split)
            ne_split['neuron_idx'] = new_idx
            for stim in ['dmr0', 'dmr1', 'spon0', 'spon1']:
                del ne_split[stim].patterns_sham
            ne_split_subsample.append(ne_split)

        savename = re.sub('0.pkl', '0-split-sub{}.pkl'.format(n_neuron), file)
        with open(savename, 'wb') as output:
            pickle.dump(ne_split_subsample, output, pickle.HIGHEST_PROTOCOL)


def get_split_ne_null_df(files, savefolder):
    ne_null = pd.DataFrame(columns=['exp', 'probe', 'region', 'stim', 'dmr_first', 'idx1', 'idx2',
                                    'pattern1', 'pattern2', 'corr', 'corr_thresh'])
    for idx, file in enumerate(files):
        print('({}/{}) get null split cNE patterns for {}'.format(idx + 1, len(files), file))
        exp = re.findall('\d{6}_\d{6}', file)[0]
        probe = re.findall('H\d{2}x\d{2}', file)[0]
        region = 'A1' if probe == 'H22x32' else 'MGB'
        with open(file, 'rb') as f:
            ne_split_null = pickle.load(f)

        for ne_split in ne_split_null:
            new_rows = get_matching_patterns_for_df(ne_split, exp, probe, region)
            ne_null = pd.concat([ne_null, new_rows], ignore_index=True)

    ne_null.to_json(os.path.join(savefolder, 'split_cNE_null.json'))


# ---------------------------------------- single unit properties -----------------------------------------------------


def calc_strf(stim_mat, spktrain, nlag=0, nlead=20, mdb=40):
    strf = np.zeros((stim_mat.shape[0], nlag + nlead))
    pp = mdb ** 2 / 8
    T = stim_mat.shape[-1] * .5 / 1e3
    for i in range(nlead):
        strf[:, i] = stim_mat @ np.roll(spktrain, i - nlead)
        
    for i in range(nlag):
        strf[:, nlead + i] = stim_mat @ np.roll(spktrain, i)
    
    strf = strf / pp / T
    return strf


def calc_crh(spktrain, stim):
    tmf = stim['sprtmf'].flatten()[spktrain > 0]
    smf = stim['sprsmf'].flatten()[spktrain > 0]

    # edges for temporal bins
    tmfaxis = stim['tmfaxis'].flatten()
    dt = tmfaxis[1] - tmfaxis[0]
    edges_tmf = np.append(tmfaxis - dt / 2, tmfaxis[-1] + dt / 2)
    # edges for spectral
    smfaxis = stim['smfaxis'].flatten()
    df = smfaxis[1] - smfaxis[0]
    edges_smf = np.append(smfaxis - df / 2, smfaxis[-1] + df / 2)
    crh, _, _ = np.histogram2d(smf, tmf, [edges_smf, edges_tmf])
    crh[0][:len(tmfaxis) // 2] = crh[0][::-1][:len(tmfaxis) // 2] / 2
    crh[0][len(tmfaxis) // 2 + 1:] = crh[0][len(tmfaxis) // 2 + 1:] / 2

    return crh, tmfaxis, smfaxis


def calc_strf_ri(spktrain, stim_mat, nlead=20, nlag=0, method='block', n_block=10, n_sample=1000, bigmat_file=None):
    strf = []
    ri = np.empty(n_sample)
    if method == 'block':
        nt = stim_mat.shape[1]
        nt_block = nt // n_block
        for i in range(n_block):
            strf.append(calc_strf(stim_mat[:, i * nt_block:(i + 1) * nt_block],
                                  spktrain[i * nt_block:(i + 1) * nt_block],
                                  nlag, nlead))
        strf = np.array(strf)
        strf_sum = np.sum(strf, axis=0)
        for i in range(n_sample):
            idx = rand.sample(range(10), 5)
            strf1 = np.sum(strf[np.array(idx), :, :], axis=0)
            strf2 = strf_sum - strf1
            ri[i] = np.corrcoef(strf1.flatten(), strf2.flatten())[0, 1]
    elif method == 'spike':
        # compared to calculate strf for each sampling (394s)
        # store strf of each spike in big mat reduce time by 90% (36s)
        # compared to calculating bigmat every time (2.4s), loading bigmat from file is faster by 40% (1.4s)
        with open(bigmat_file, 'rb') as f:
            bigmat = pickle.load(f)

        strf_mat = bigmat[:, spktrain > 0]
        strf = np.sum(strf_mat, axis=1)

        # compared to get strf for each sampling (36s), matrix multiplication reduce time by 50% (17s)
        nspk = strf_mat.shape[1]
        spk1 = np.empty([strf_mat.shape[1], n_sample])
        for i in range(n_sample):
            idx = rand.sample(range(nspk), nspk // 2)
            spk1[idx, i] = 1
        strf1 = np.transpose(strf_mat @ spk1)
        strf2 = np.tile(strf, (n_sample, 1)) - strf1
        ri = np.corrcoef(strf1, strf2)[:n_sample, n_sample:].diagonal()
    return ri


def calc_crh_ri(spktrain, stim, method='block', n_block=10, n_sample=1000):
    crh = []
    ri = np.empty(n_sample)

    if method == 'block':
        nt = stim['sprtmf'].flatten().shape[0]
        nt_block = nt // n_block
        for i in range(n_block):
            spktrain_tmp = np.empty(spktrain.shape)
            spktrain_tmp[i * nt_block: (i + 1) * nt_block] = spktrain[i * nt_block: (i + 1) * nt_block]
            crh_tmp, _, _ = calc_crh(spktrain_tmp, stim)
            crh.append(crh_tmp)
        crh = np.array(crh)
        crh_sum = np.sum(crh, axis=0)
        for i in range(n_sample):
            idx = rand.sample(range(10), 5)
            crh1 = np.sum(crh[np.array(idx), :, :], axis=0)
            crh2 = crh_sum - crh1
            ri[i] = np.corrcoef(crh1.flatten(), crh2.flatten())[0, 1]
    elif method == 'spike':
        crh, _, _ = calc_crh(spktrain, stim)
        spk_idx = np.where(spktrain > 0)[0]
        nspk = spk_idx.shape[0]
        for i in range(n_sample):
            idx = rand.sample(range(nspk), nspk // 2)
            spktrain_tmp = np.empty(spktrain.shape)
            spktrain_tmp[spk_idx[idx]] = 1
            crh1, _, _ = calc_crh(spktrain_tmp, stim)
            crh2 = crh - crh1
            ri[i] = np.corrcoef(crh1.flatten(), crh2.flatten())[0, 1]
    return np.array(ri)


def calc_strf_properties(strf, taxis, faxis, sigma_y=2, sigma_x=1):
    if strf.ndim == 3:
        strf = np.sum(strf, axis=0)
    idx_t = np.where(taxis < 50)[0]
    strf = strf[:, idx_t]
    taxis = taxis[idx_t]
    weights = np.array([[1],
                        [2],
                        [1]],
                       dtype=np.float)
    weights = weights / np.sum(weights[:])
    strf_filtered = convolve(strf, weights, mode='constant')
    row, col = np.unravel_index(abs(strf_filtered).argmax(), strf.shape)
    bf = faxis[row]
    latency = taxis[col]
    return bf, latency


def calc_crh_properties(crh, tmfaxis, smfaxis):
    tmf = np.sum(crh, axis=0)
    smf = np.sum(crh, axis=1)

    # upsample tmf and smf
    tmf_u = scipy.interpolate.interp1d(tmfaxis, tmf, kind='cubic')
    smf_u = scipy.interpolate.interp1d(smfaxis, smf, kind='cubic')
    x = np.linspace(tmfaxis[0], tmfaxis[-1], 1000)
    tmf_u = tmf_u(x)
    btmf = x[tmf_u.argmax()]
    x = np.linspace(smfaxis[0], smfaxis[-1], 1000)
    smf_u = smf_u(x)
    bsmf = x[smf_u.argmax()]

    return btmf, bsmf


def calc_strf_ptd(strf, nspk):
    return (strf.max() - strf.min()) / nspk


def moran_i(mat, diagopt=True):
    weightmat = create_spatial_autocorr_weight_mat(*mat.shape, diagopt=diagopt)
    N = mat.size
    W = weightmat.sum()

    # Calculate denominator
    xbar = mat.mean()
    dvec = mat - xbar
    dvec = dvec.flatten()
    deno = np.multiply(dvec, dvec).sum()

    # Calculate numerator
    numer = 0
    i, j = np.where(weightmat)

    for m, n in zip(i, j):
        numer += dvec[m] * dvec[n]
    return (N / W) * (numer / deno)


def create_spatial_autocorr_weight_mat(nrows, ncols, diagopt=True):
    eq = [lambda i, j: np.array([i - 1, j]),
          lambda i, j: np.array([i + 1, j]),
          lambda i, j: np.array([i, j - 1]),
          lambda i, j: np.array([i, j + 1])]
    if diagopt:
        eq.extend([lambda i, j: np.array([i - 1, j - 1]),
                   lambda i, j: np.array([i + 1, j - 1]),
                   lambda i, j: np.array([i - 1, j + 1]),
                   lambda i, j: np.array([i + 1, j + 1])])

    nele = nrows * ncols;
    weightmat = np.zeros((nele, nele))
    # single index

    for j in range(ncols):
        for i in range(nrows):
            # apply all equations to find 2d-indices of contiguous pixels
            rcidx = [func(i, j) for func in eq]
            # keep only indices within range
            rcidx = [[x, y] for [x, y] in rcidx if x >= 0 and x < nrows and y >= 0 and y < ncols]
            # get 1d-indices from 2d
            idx = [np.ravel_multi_index(x, [nrows, ncols]) for x in rcidx]

            weightmat[i * nrows + j, idx] = 1
    return weightmat


def calc_strf_nonlinearity(strf, spktrain, stim):
    if strf.ndim == 3:
        strf = np.sum(strf, axis=0)
    ntbins = strf.shape[-1]
    similarity_null = get_strf_proj_xprior(strf, stim)

    sd = similarity_null.std()
    m = similarity_null.mean()
    similarity_null = (similarity_null - m) / sd

    edge_min = np.round(similarity_null.min()) - .5
    edge_max = np.round(similarity_null.max()) + .5
    edges = np.arange(edge_min, edge_max + 0.5)
    centers = (edges[:-1] + edges[1:]) / 2
    t, _ = np.histogram(similarity_null, edges)
    t = t * 5e-3  # time of each sd bin in s

    similarity_spk = similarity_null[spktrain[ntbins - 1:] > 0]
    nspk, _ = np.histogram(similarity_spk, edges)

    fr = np.divide(nspk, t)
    fr_mean = nspk.sum() / t.sum()

    idx0 = np.where(centers == 0)[0][0]
    fr_l = fr[:idx0].sum()
    fr_r = fr[idx0 + 1:].sum()
    asi = (fr_r - fr_l) / (fr_r + fr_l)
    nonlinearity = {'si_null': similarity_null, 'si_spk': similarity_spk,
                    'centers': centers, 't_bins': t, 'nspk_bins': nspk,
                    'fr': fr, 'fr_mean': fr_mean, 'asi': asi}

    return nonlinearity


def calc_strf_mi(spiketimes, edges, stim, frac=[90, 92.5, 95, 97.5, 100], nreps=10, nblocks=2):
    spktrain, _ = np.histogram(spiketimes, edges)
    spiketimes = spiketimes[(spiketimes >= edges[0]) & (spiketimes <= edges[-1])]
    np.random.shuffle(spiketimes)
    nspk_block = int(spktrain.sum() / nblocks)
    info = np.zeros(nblocks)
    ifrac = np.zeros([nblocks, nreps, len(frac)])
    xbins_centers = np.zeros([nblocks, 14])

    for i in range(nblocks):
        spiketimes_tmp = spiketimes[i * nspk_block: (i + 1) * nspk_block]
        spktrain_train, _ = np.histogram(spiketimes_tmp, edges)
        spktrain_test = spktrain - spktrain_train
        info[i], ifrac[i], xbins_centers[i] = cal_mi(spktrain_train, spktrain_test, stim)

    return info, ifrac, xbins_centers


def cal_mi(spktrain_train, spktrain_test, stim, frac=[90, 92.5, 95, 97.5, 100], 
           nreps=10, nlag=0, nlead=20):
    strf = calc_strf(stim, spktrain_train, nlag=nlag, nlead=nlead)
    xprior = get_strf_proj_xprior(strf, stim)
    if np.sum(spktrain_test) == 0: # only one block
        xpost = xprior[spktrain_train[(nlead - 1):] > 0]
    else:
        xpost = xprior[spktrain_test[(nlead - 1):] > 0]

    ifrac, xbins_centers = info_from_fraction(xprior, xpost, frac=frac, nreps=nreps)

    info_mn = np.mean(ifrac, axis=0)
    x = 100 / np.array(frac)

    _, info = np.polyfit(x, info_mn, 1)

    return info, ifrac, xbins_centers


def info_from_fraction(xprior, xpost, frac=[90, 92.5, 95, 97.5, 100], nreps=10):
    ifrac = np.empty([nreps, len(frac)])
    n_event = len(xpost)

    # prior distribution of projection values
    xmn = xprior.mean()
    xstd = xprior.std()
    xprior = (xprior - xmn) / xstd
    xpost = (xpost - xmn) / xstd
    xbins_edges = np.linspace(xprior.min(), xprior.max(), 15)
    xbins_centers = (xbins_edges[:-1] + xbins_edges[1:]) / 2
    nx_prior, _ = np.histogram(xprior, xbins_edges)
    px_prior = nx_prior / len(xprior)

    # posterior distribution of projection values of subset of spikes
    for m, curr_frac in enumerate(frac):
        n_event_subset = int(np.round(curr_frac / 100 * n_event))

        if curr_frac == 100:
            nreps = 1

        for n in range(nreps):
            xspk = rand.sample(list(xpost), n_event_subset)
            nx_post, _ = np.histogram(xspk, xbins_edges)
            px_post = nx_post / n_event_subset
            ifrac[n, m] = info_prior_post(px_prior, px_post)

        if curr_frac == 100:
            ifrac[:, m] = ifrac[n, m]

    return ifrac, xbins_centers


def get_strf_proj_xprior(strf, stim):
    """
    get the projection value of strt and the entire stimulus (xprior for info/nonlinearity calculation)

    Parameters
    ----------
    strf : np.array, m * n
        spectrotemporal receptive field  (STRF) of the unit
    stim : np.array, nf * nt
        spectrogram of the stimulus

    Returns
    -------
    similarity_null : np.array, 1 * (nt - n + 1)
        xprior. projection value of strf with the entire stimulus

    """
    ntbins = strf.shape[1]
    similarity_tbins = np.transpose(strf) @ stim
    similarity_null = np.zeros(stim.shape[1] - ntbins + 1)
    for i in range(ntbins):
        if i == ntbins - 1:
            similarity_null = similarity_null + similarity_tbins[i, i:]
        else:
            similarity_null = similarity_null + similarity_tbins[i, i:-ntbins + i + 1]

    return similarity_null


def info_prior_post(px_prior, px_post):
    idx = np.where((px_prior > 0) & (px_post > 0))[0]
    px_prior = px_prior[idx]
    px_post = px_post[idx]
    return np.sum(px_post * np.log2(px_post / px_prior))


def ICweight_match_binsize_adjacent(datafolder, file, dfs):
    base = re.findall('.*db', file)[0]
    suffix = re.findall('dft-.*', file)[0]

    patterns = []
    for i, df in enumerate(dfs):
        file = glob.glob('{}-*{}{}'.format(base, df, suffix))[0]
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        patterns.append(ne.patterns)

    n_ne = [len(x) for x in patterns]
    p = np.ones([n_ne[0], len(dfs) - 1])
    corr = np.zeros([n_ne[0], len(dfs) - 1])
    for i in range(len(n_ne) - 1):
        pattern1 = np.array(patterns[i])
        row_idx = np.where(abs(pattern1).sum(axis=1) > 0)[0]  # get index of existing patterns
        pattern2 = np.array(patterns[i + 1])
        patterns[i + 1] = np.zeros([n_ne[0], pattern1.shape[1]])
        if not any(row_idx):
            continue
        pattern1 = pattern1[row_idx, :]

        corrmat = np.corrcoef(pattern1, pattern2)[:len(row_idx), len(row_idx):]
        for _ in range(corrmat.shape[0]):
            row, col = np.unravel_index(abs(corrmat).argmax(), corrmat.shape)
            corrmat[row, :] = 0
            corrmat[:, col] = 0
            corr[row_idx[row], i], p[row_idx[row], i] = pearsonr(pattern1[row], pattern2[col])
            patterns[i + 1][row_idx[row]] = pattern2[col]

    return {'df': dfs, 'n_ne': n_ne, 'patterns': patterns, 'pearsonr': corr, 'p': p}


def ICweight_match_binsize(datafolder, file, dfs):
    base = re.findall('.*fs20000', file)[0]
    suffix = re.findall('dft-.*', file)[0]

    with open(file, 'rb') as f:
        ne = pickle.load(f)
    patterns_ref = ne.patterns
    patterns_match = np.zeros([patterns_ref.shape[0], len(dfs), patterns_ref.shape[1]])
    corr_match = np.zeros([patterns_ref.shape[0], len(dfs)])
    for i, df in enumerate(dfs):
        file = glob.glob('{}-*-{}{}'.format(base, df, suffix))[0]
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        patterns = ne.patterns
        for j, pattern_ref in enumerate(patterns_ref):
            corr = np.corrcoef(pattern_ref, patterns)[0][1:]
            match_idx = np.argmax(abs(corr))
            pattern = patterns[match_idx]
            patterns_match[j, i, :] = pattern
            corr_match[j, i] = abs(corr[match_idx])

    return {'df': dfs, 'patterns': patterns_match, 'pearsonr': corr_match}


def batch_save_icweight_binsize_corr_to_dataframe(datafolder, savefolder, dfs):
    dataframe = pd.DataFrame()
    for stim in ('spon', 'dmr'):
        files = glob.glob(datafolder + r'/{}/*-20dft-{}.pkl'.format(stim, stim), recursive=False)
        for idx, file in enumerate(files):
            print('({}/{}) save icweight binsize match summary data for {}'.format(idx, len(files), file))
            dataframe_tmp = save_icweight_binsize_corr_to_dataframe(datafolder, savefolder, file, dfs)
            dataframe_tmp['stim'] = stim
            dataframe = pd.concat([dataframe, dataframe_tmp])
    dataframe.reset_index(inplace=True, drop=True)
    dataframe.to_json(os.path.join(savefolder, 'icweight_corr_binsize.json'))


def save_icweight_binsize_corr_to_dataframe(datafolder, savefolder, file, dfs):
    dataframe = pd.DataFrame()
    base = re.findall('.*fs20000', file)[0]
    suffix = re.findall('dft-.*', file)[0]
    exp = re.search('\d{6}_\d{6}', file).group(0)
    probe = re.search('H\d{2}x\d{2}', file).group(0)
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    patterns_ref = ne.patterns
    thresh = 1 / np.sqrt(patterns_ref.shape[1])

    for i, df in enumerate(dfs):
        file = glob.glob('{}-*-{}{}'.format(base, df, suffix))[0]
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        patterns = ne.patterns
        for j, pattern in enumerate(patterns):
            corr = np.corrcoef(pattern, patterns_ref)[0][1:]
            match_idx = np.argmax(abs(corr))
            pattern_ref = patterns_ref[match_idx]

            members_ref = set(np.where(pattern_ref > thresh)[0])
            n_member_ref = len(members_ref)
            members = set(np.where(pattern > thresh)[0])
            n_member = len(members)
            member_overlap = members.intersection(members_ref)
            n_member_overlap = len(member_overlap)
            member_ref_only = members_ref.difference(members)
            member_extra = members.difference(members_ref)
            n_member_all = len(members.union(members_ref))
            row = pd.DataFrame({'exp': exp, 'probe': probe, 'df': df, 'corr': abs(corr[match_idx]),
                          'member_overlap': [member_overlap], 'member_extra': [member_extra],
                          'member_ref_only': [member_ref_only], 'n_member': n_member, 'n_member_ref': n_member_ref,
                          'n_member_all': n_member_all, 'n_member_overlap': n_member_overlap})
            dataframe = pd.concat([dataframe, row])
    return dataframe

    return {'df': dfs, 'patterns': patterns_match, 'pearsonr': corr_match}


def batch_save_icweight_ccg_binsize(datafolder, jsonfile, savefolder):
    data = pd.read_json(jsonfile)
    data['overlap_prc'] = data['n_member_overlap'] / data['n_member_ref']
    data = data[data.overlap_prc > .5]
    ccg = pd.DataFrame()
    for stim in ('spon', 'dmr'):
        for probe in ('H31x64', 'H22x32'):
            data_subset = data[(data.stim == stim) & (data.probe == probe)]
            exps = set(data_subset.exp)
            dfs = set(data_subset.df)
            for idx, exp in enumerate(exps):
                print('{}/{}'.format(idx+1, len(exps)), stim, probe, exp)
                exp_str = str(exp)
                exp_str = exp_str[:6] + '_' + exp_str[6:]
                session_file = glob.glob(os.path.join(datafolder, f'{exp_str}-*-{probe}-fs20000.pkl'))
                assert(len(session_file) == 1)
                with open(session_file[0], 'rb') as f:
                    session = pickle.load(f)
                spktrain, edges = session.downsample_spktrain(df=2, stim=stim)
                for df in dfs:
                    cnes = data_subset[(data_subset.df == df) & (data_subset.exp == exp)]
                    member_pairs_overlap = set()
                    member_pairs_ref = set()
                    member_pairs_extra = set()
                    for i in range(len(cnes)):
                        cne = cnes.iloc[i]
                        members_overlap = cne.member_overlap
                        members_ref_only = cne.member_ref_only
                        members_extra = cne.member_extra
                        member_pairs_overlap.update(list(itertools.combinations(members_overlap, 2)))
                        member_pairs_ref.update([(x, y) for x in members_ref_only for y in members_overlap])
                        member_pairs_extra.update([(x, y) for x in members_extra for y in members_overlap])
                    member_pairs_overlap = set([(x, y) if x < y else (y, x) for x, y in member_pairs_overlap])
                    member_pairs_ref = set([(x, y) if x < y else (y, x) for x, y in member_pairs_ref])
                    member_pairs_extra = set([(x, y) if x < y else (y, x) for x, y in member_pairs_extra])
                    member_pairs_ref = member_pairs_ref.difference(member_pairs_overlap)
                    member_pairs_extra = member_pairs_extra.difference(members_overlap)
                    ccg_overlap = get_ccg(spktrain,  member_pairs_overlap)
                    ccg_ref = get_ccg(spktrain, member_pairs_ref)
                    ccg_extra = get_ccg(spktrain,  member_pairs_extra)
                    ccg = pd.concat([ccg, pd.DataFrame({'exp': exp, 'stim': stim, 'probe': probe, 'df': df,
                                                        'ccg_overlap': [ccg_overlap],
                                                        'ccg_ref': [ccg_ref], 'ccg_extra': [ccg_extra]})])
            ccg.reset_index(inplace=True, drop=True)
            ccg.to_json(os.path.join(savefolder, 'member_pair_ccg_binsize.json'))


def get_ccg(spktrain, pairs, hw=200):
    ccg = []
    for i, pair in enumerate(pairs):
        ccg.append(np.correlate(spktrain[pair[0]], spktrain[pair[1]][hw:-hw]))
    return np.array(ccg)


def shuffle_spktrain(spktrain):
    for i in range(spktrain.shape[0]):
        shift = rand.randint(1, spktrain.shape[1])
        spktrain[i] = np.roll(spktrain[i], shift)
    return spktrain


def get_num_cne_vs_shuffle(datafolder='E:\Congcong\Documents\data\comparison\data-pkl',
                           summary_folder='E:\Congcong\Documents\data\comparison\data-summary'):
    stims = ('dmr', 'spon')
    n_ne = {'exp': [], 'n_ne': [], 'shuffled': [], 'stim': [], 'region': []}
    for stim in stims:
        nefiles = glob.glob(os.path.join(datafolder, '*-ne-20dft-{}.pkl'.format(stim)))
        for file_idx, file in enumerate(nefiles):
            print('{}/{} get ne numbers for {}'.format(file_idx + 1, len(nefiles), file))
            if 'H31x64' in file:
                n_ne['region'].append('MGB')
            else:
                n_ne['region'].append('A1')
            # get number of cNEs from real data
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            n_ne['exp'].append(ne.exp)
            n_ne['n_ne'].append(len(ne.ne_members))
            n_ne['stim'].append(stim)
            # get number of cNEs from shuffled
            file_null = re.sub('{}.pkl'.format(stim), '{}-shuffled.pkl'.format(stim), file)
            with open(file_null, 'rb') as f:
                data = pickle.load(f)
            n_ne['shuffled'].append(np.median(data['n_ne']))

    n_ne = pd.DataFrame(n_ne)
    n_ne.to_json(os.path.join(summary_folder, 'num_ne_data_vs_shuffle.json'))
