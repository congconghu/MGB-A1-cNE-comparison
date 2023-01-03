import re
import glob
import pickle
import numpy as np
from itertools import combinations
from scipy.stats import zscore

import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA


def detect_cell_assemblies(spktrain):
    spktrain_z = zscore(spktrain, 1)
    pca = PCA()
    pca.fit(np.transpose(spktrain_z))

    # get threshold for significant PCs (based on Marcenko-Patur distribution)
    thresh = get_pc_thresh(spktrain)
    num_ne = sum(pca.explained_variance_ > thresh)
    patterns = fast_ica(spktrain_z, num_ne)
    return normalize_pattern(patterns)


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


def calc_strf(stim_mat, spktrain, nlag, nlead):
    strf = np.zeros((stim_mat.shape[0], nlag + nlead))
    for i in range(nlead):
        strf[:, i] = stim_mat @ np.roll(spktrain, i - nlead)
    return strf


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
    return ne_split

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
        corr_dmr_spon[:i+1, :] = 0
        corr_dmr_spon[:, :i+1] = 0
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
    pass