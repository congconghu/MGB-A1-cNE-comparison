import re
import glob
import pickle
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
    xcorr = {'xcorr': [], 'member': [], 'stim': [], 'region': []}
    for idx, file in enumerate(files):
        print('({}/{}) get member and nonmmebers xcorr for {}'.format(idx + 1, len(files), file))

        with open(file, 'rb') as f:
            session = pickle.load(f)

        n_neuron = len(session.units)
        all_pairs = set(combinations(range(n_neuron), 2))

        ne_file_path = re.sub('fs20000', 'fs20000-ne-20dft*', session.file_path)
        nefiles = glob.glob(ne_file_path)
        for nefile in nefiles:

            # get region of the recording
            if session.depth > 2000:
                region = 'MGB'
            else:
                region = 'A1'
            with open(nefile, 'rb') as f:
                ne = pickle.load(f)

            # get stimulus condition of the cNEs
            if 'dmr' in nefile:
                stim = 'dmr'
            else:
                stim = 'spon'

            member_pairs = set()
            for members in ne.ne_members.values():
                member_pairs.update(set(combinations(members, 2)))
            nonmember_pairs = all_pairs.difference(member_pairs)

            spktrain, _ = session.downsample_spktrain(df=df, stim=stim)
            spktrain_shift = np.roll(spktrain, -maxlag, axis=1)
            spktrain_shift = spktrain_shift[:, :-2*maxlag]
            for u1, u2 in member_pairs:
                c = np.correlate(spktrain[u1], spktrain_shift[u2], mode='valid')
                xcorr['xcorr'].append(np.array(c).astype('int16'))
                xcorr['member'].append(True)
                xcorr['stim'].append(stim == 'dmr')
                xcorr['region'].append(region)
            for u1, u2 in nonmember_pairs:
                c = np.correlate(spktrain[u1], spktrain_shift[u2], mode='valid')
                xcorr['xcorr'].append(c)
                xcorr['member'].append(False)
                xcorr['stim'].append( stim == 'dmr')
                xcorr['region'].append(region)
    xcorr = pd.DataFrame(xcorr)
    return xcorr