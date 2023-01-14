import re
import glob
import os
import pickle
import random as rand
from itertools import combinations
from scipy.stats import zscore
from copy import deepcopy

import numpy as np
import pandas as pd

from scipy.ndimage.filters import convolve
from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from helper import Timer


def detect_cell_assemblies(spktrain):
    spktrain_z = zscore(spktrain, 1)
    pca = PCA()
    pca.fit(np.transpose(spktrain_z))

    # get threshold for significant PCs (based on Marcenko-Patur distribution)
    thresh = get_pc_thresh(spktrain)
    num_ne = sum(pca.explained_variance_ > thresh)
    patterns = fast_ica(spktrain_z, num_ne)
    return normalize_pattern(patterns)


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
    nbins = len(edges) - 1
    dig = np.digitize(spiketimes, edges)
    idx_bin, idx_spiketimes = np.unique(dig, return_index=True)
    spiketimes_binned = [[] for i in range(nbins)]
    for i, ibin in enumerate(idx_bin):
        if 0 < ibin <= nbins:
            try:
                spiketimes_binned[ibin - 1] = spiketimes[idx_spiketimes[i]: idx_spiketimes[i + 1]]
            except IndexError:  # when there is no spike after the last bin
                spiketimes_binned[ibin - 1] = spiketimes[idx_spiketimes[i]:]
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

            member_pairs = set()
            for members in ne.ne_members.values():
                member_pairs.update(set(combinations(members, 2)))
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
    return corr_mat, [order1, order2], [patterns1[np.array(order1)], patterns2[np.array(order2)]]


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
                                        'pattern1': [ne_split[stim+'0'].patterns[idx1]],
                                        'pattern2': [ne_split[stim+'1'].patterns[idx2]],
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

    ne.to_json(os.path.join(savefolder, 'split_cNE.json'))


def sub_sample_split_ne(files, savefolder, n_neuron=10, n_sample=10):
    rand.seed(0)
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            session = pickle.load(f)

        if len(session.units) <= n_neuron:
            continue
        elif not hasattr(session, 'spktrain_spon'):
            continue

        print('({}/{}) get split cNEs {}'.format(i+1, len(files), file))
        spktrain_dmr = deepcopy(session.spktrain_dmr)
        spktrain_spon = deepcopy(session.spktrain_spon)

        ne_split_subsample = []
        for sample in range(n_sample):
            print('{}/{}'.format(sample+1, n_sample))
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


# ---------------------------------------- single unit properties -----------------------------------------------------


def calc_strf(stim_mat, spktrain, nlag, nlead):
    strf = np.zeros((stim_mat.shape[0], nlag + nlead))
    for i in range(nlead):
        strf[:, i] = stim_mat @ np.roll(spktrain, i - nlead)
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
    crh[0][:len(tmfaxis)//2] = crh[0][::-1][:len(tmfaxis)//2]/2
    crh[0][len(tmfaxis) // 2 + 1:] =  crh[0][len(tmfaxis) // 2 + 1:] / 2

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
            spktrain_tmp[i * nt_block: (i + 1) * nt_block] =  spktrain[i * nt_block : (i + 1) * nt_block]
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