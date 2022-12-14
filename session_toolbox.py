import pickle
import re
from scipy.stats import zscore
import random

import mat73
import numpy as np
from scipy.io import loadmat
import ne_toolbox as netools
from itertools import combinations

from helper import strf2rtf


class Stimulus:

    def __init__(self, stim_mat=None, taxis=None, faxis=None, phase=None, fs=None, df=None, app=None, amp_dist=None,
                 max_fm=None, max_rd=None):
        """Initiate empty object of Stimulus (can be later modified to create Stimulus from scratch)

        INPUT:
        stim_mat: nf x nt spectrogram of stimulus
        taxis: time axis for a block of stimulus
        faxis: frequency axis of the prectrogram
        phase: phase at each time bin for the spectrogram
        fs: sound file sampling rate, in Hz
        df: down sample rate to get stim_mat from the sound file
        app: modulation amplitude
        amp_dist: unit of modulation amplitude
        max_fm: maximum frequency modulation rate
        max_rd: maximum temporal modulation rate
        """
        self.stim_mat = stim_mat
        self.taxis = taxis
        self.faxis = faxis
        self.phase = phase
        self.fs = fs
        self.DF = df
        self.App = app
        self.AmpDist = amp_dist
        self.MaxFM = max_fm
        self.MaxRD = max_rd

    def read_mat_file(self, stimfile_path):
        """Read .mat files for the stimulus and save parameters in Stimulus class"""
        # load stimulus file
        data_dict = mat73.loadmat(stimfile_path)
        self.stim_mat = data_dict['stim_mat']
        # load stimulus parameters
        paramfile_path = re.sub('_stim', '_param', stimfile_path)
        params = loadmat(paramfile_path)
        self.taxis = params['taxis'][0]
        self.faxis = params['faxis'][0]
        self.phase = params['phase'][0]
        self.fs = params['Fs'][0][0] * 1e3
        self.DF = params['DF'][0][0]
        self.App = params['App'][0][0]
        self.AmpDist = params['AmpDist'][0]
        self.MaxFM = params['MaxFM'][0][0]
        self.MaxRD = params['MaxRD'][0][0]

    def save_pkl_file(self, savefile_path):
        with open(savefile_path, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    def down_sample(self, df=10):
        self.stim_mat = self.stim_mat[:, (df-1)::df]
        self.df = df
        self.taxis = self.taxis[::10]

    def mtf(self, nleads=200, tmodbins=8, smodbins=16):
        """
        get modulation transfer function (mtf) of the stimulus
        :param nleads: number of time points to consider when getting modulation rate
        :param tmodbins: number of bins for temporal modulation rate
        :param smodbins: number of bins for spectral modulation rate
        :return: dictionary containing mtf information
            tmf: temporal modulation frequency
            smf: spectral modulation frequency
            tmf_axis: tmf axis
            smf_axis: smf axis
        """
        print('Get mtf for stimulus')
        nt = self.stim_mat.shape[1]
        tmf = np.empty(nt)
        smf = np.empty(nt)
        for i in range(nleads, nt+1):
            if i % 1000 == 0:
                print('{}/{}'.format(i, nt))
            frame = self.stim_mat[:, i - nleads: i]
            tmf_axis, smf_axis, rtf = strf2rtf(frame, taxis=self.taxis, faxis=self.faxis,
                                               maxtm=self.MaxFM, maxsm=self.MaxRD,
                                               tmodbins=tmodbins, smodbins=smodbins)
            idx_fmax, idx_tmax = np.unravel_index(rtf.argmax(), rtf.shape)
            smf[i] = smf_axis[idx_fmax]
            tmf[i] = tmf_axis[idx_tmax]

        stim = {'tmf': tmf, 'smf': smf, 'tmf_axis': tmf_axis, 'smf_axis': smf_axis}
        return stim


class SingleUnit:

    def __init__(self, unit, spiketimes):
        self.unit = unit
        self.spiketimes = spiketimes

    def add_properties(self, chan=None, isolation=None, peak_snr=None, isi_bin=None, isi_vio=None,
                       waveforms=None, waveforms_mean=None, waveforms_std=None, amps=None,
                       adjacent_chan=None, template=None):
        self.chan = chan
        self.isolation = isolation
        self.peak_snr = peak_snr
        self.isi_bin = isi_bin
        self.isi_vio = isi_vio
        self.waveforms = waveforms
        self.waveforms_mean = waveforms_mean
        self.waveforms_std = waveforms_std
        self.amps = amps
        self.adjacent_chan = adjacent_chan
        self.template = template

    def get_strf(self, stim, edges, nlead=20, nlag=0):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)
        stim_mat = stim.stim_mat
        taxis = (stim.taxis[1] - stim.taxis[0]) * np.array(range(nlead-1, -nlag-1, -1)) * 1000
        faxis = stim.faxis
        strf = netools.calc_strf(stim_mat, spktrain, nlag=nlag, nlead=nlead)
        self.strf = strf
        self.strf_taxis = taxis
        self.strf_faxis = faxis

    def get_crh(self, stim, edges):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)
        self.crh, self.tmfaxis, self.smfaxis = netools.calc_crh(spktrain, stim)

    def get_strf_ri(self, stim, edges, nlead=20, nlag=0, method='block', n_block=10, n_sample=1000):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)
        stim_mat = stim.stim_mat
        taxis = (stim.taxis[1] - stim.taxis[0]) * np.array(range(nlead - 1, -nlag - 1, -1)) * 1000
        faxis = stim.faxis

        ri = netools.calc_strf_ri(spktrain, stim_mat, nlead=nlead, nlag=nlag, method=method,
                                      n_block=n_block, n_sample=n_sample, bigmat_file=stim.bigmat_file)
        self.strf_ri = ri

        spktrain_null = spktrain[::-1]
        ri_null = netools.calc_strf_ri(spktrain_null, stim_mat, nlead=nlead, nlag=nlag, method=method,
                                           n_block=n_block, n_sample=n_sample, bigmat_file=stim.bigmat_file)
        self.strf_ri_null = ri_null
        self.strf_ri_z = (ri.mean() - ri_null.mean()) / np.sqrt((ri.std() ** 2 + ri_null.std() ** 2) / 2)
        self.strf_ri_p = sum(ri_null > ri.mean()) / n_sample

    def get_crh_ri(self, stim, edges, method='block', n_block=10, n_sample=1000):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)

        ri = netools.calc_crh_ri(spktrain, stim, method=method, n_block=n_block, n_sample=n_sample)
        self.crh_ri = ri

        spktrain_null = spktrain[::-1]
        ri_null = netools.calc_crh_ri(spktrain_null, stim, method=method, n_block=n_block, n_sample=n_sample)
        self.crh_ri_null = ri_null
        self.crh_ri_z = (ri.mean() - ri_null.mean()) / np.sqrt((ri.std() ** 2 + ri_null.std() ** 2) / 2)
        self.crh_ri_p = sum(ri_null > ri.mean()) / n_sample



class Session:

    def __init__(self, exp=None, site=None, depth=None, filt_params=None, fs=None, probe=None, probdata=None, units=[],
                 trigger=None):
        """Initiate empty object of Stimulus (can be later modified to create Stimulus from scratch)

        INPUT:
        exp: time of recording in the format of "YYMMDD_HHMMSS"
        site: recording site
        depth: depth of the tip of probe
        filt_params: filter parameter for automatic curation
        fs: recording sampling rate in Hz
        probe: type of probe used for recording, 'H31x64' or 'H22x32'
        probdata: layout of probe
        """
        self.exp = exp
        self.site = site
        self.depth = depth
        self.filt_params = filt_params
        self.fs = fs
        self.probe = probe
        self.probdata = probdata
        self.units = units
        self.trigger = trigger

    def read_mat_file(self, datafile_path):
        """Read .mat files of the recording and convert data to Session object"""
        # load stimulus file
        data_dict = mat73.loadmat(datafile_path)
        self.exp = data_dict['spk']['exp']
        self.site = int(data_dict['spk']['site'])
        self.depth = int(data_dict['spk']['depth'])
        self.filt_params = data_dict['spk']['filt_params']
        self.fs = int(data_dict['spk']['fs'])
        self.probe = data_dict['spk']['probe']
        self.probdata = data_dict['spk']['probdata']
        self.trigger = data_dict['trigger']
        self.units = []
        for unit in data_dict['spk']['spk']:
            su = SingleUnit(unit=int(unit['unit']), spiketimes=unit['spiketimes'])
            su.add_properties(chan=int(unit['chan']), isolation=float(unit['isolation']), peak_snr=unit['peak_snr'],
                              isi_bin=float(unit['isibin']), isi_vio=float(unit['isi_vio']), template=unit['template'],
                              waveforms=unit['waveforms'], waveforms_mean=unit['waveforms_mean'],
                              waveforms_std=unit['waveforms_std'], amps=unit['amps'],
                              adjacent_chan=unit['adjacent_chan'])
            self.units.append(su)

    def save_pkl_file(self, savefile_path):
        self.file_path = savefile_path
        with open(savefile_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_spktrain_from_stim(self, stim_len):
        trigger = self.trigger
        trigger_ms = trigger / self.fs * 1e3
        nt = int(stim_len / len(trigger_ms))  # number of time points between 2 triggers
        trigger_ms = np.append(trigger_ms, [trigger_ms[-1] + max(np.diff(trigger_ms))])
        edges = np.array(
            [np.linspace(trigger_ms[i], trigger_ms[i + 1], nt + 1)[:-1] for i in range(len(trigger_ms) - 1)])
        edges = edges.flatten()
        edges = np.append(edges, trigger_ms[-1])

        # get spktrain under dmr
        spktrain = np.zeros([len(self.units), stim_len], 'int8')
        for idx, unit in enumerate(self.units):
            spktrain[idx], _ = np.histogram(unit.spiketimes, edges)
        spktrain[spktrain > 0] = 1
        self.spktrain_dmr = spktrain
        self.edges_dmr = edges

        # get spktrain under spon
        if trigger_ms[0] / 1e3 / 60 > 5:  # first trigger happens 5minutes after the start of recording -> spon first
            self.dmr_first = False
            edges = np.arange(0, trigger_ms[0], 0.5)
        else:
            self.dmr_first = True
            spiketimes_max = max([unit.spiketimes[-1] for unit in self.units])
            edges = np.arange(trigger_ms[-1], spiketimes_max + 0.5, 0.5)

        if len(
                edges) * 0.5 / 1e3 / 60 > 5:  # save spon train when recording of spontanous activity lasts more than 5 minutes
            spktrain = np.zeros([len(self.units), len(edges) - 1], 'int8')
            for idx, unit in enumerate(self.units):
                spktrain[idx], _ = np.histogram(unit.spiketimes, edges)
            spktrain[spktrain > 0] = 1
            self.spktrain_spon = spktrain
            self.edges_spon = edges
        self.save_pkl_file(self.file_path)

    def downsample_spktrain(self, df=20, stim=None):
        """down sample the time resolution of spike trains
        input:
        df: down sample factor
        stim: 'dmr' or 'spon'
        """
        try:
            spktrain = eval('self.spktrain_{}'.format(stim))
            edges = eval('self.edges_{}'.format(stim))
        except AttributeError:
            return np.array([]), np.array([])
        # down sample spktrain
        nt = spktrain.shape[1] // df
        spktrain = spktrain[:, :nt * df]
        edges = edges[0:nt * df + 1:df]
        spktrain = np.resize(spktrain, (spktrain.shape[0], nt, df))
        return np.sum(spktrain, axis=2), edges

    def get_ne(self, df, stim):
        """get cNEs in each recording session, under dmr and spon"""
        # get cNE for dmr-evoked activity
        spktrain, edges = self.downsample_spktrain(df, stim)
        if spktrain.any():
            patterns = netools.detect_cell_assemblies(spktrain)
            ne = NE(self.exp, self.depth, self.probe, df, stim=stim, spktrain=spktrain, patterns=patterns, edges=edges)
            return ne
        else:
            return None

    def get_ne_split(self, df):
        """split sensory-evoked and spontaneous activities in 2 blocks respectively and get cNEs on each block"""
        ne_split = dict()
        if hasattr(self, 'spktrain_dmr') and hasattr(self, 'spktrain_spon'):
            stims = ('dmr', 'spon')
            for stim in stims:
                spktrain, edges = self.downsample_spktrain(df, stim)
                midpoint = spktrain.shape[1] // 2
                spktrains = [spktrain[:, :midpoint], spktrain[:, midpoint:]]
                edges = [edges[:midpoint+1], edges[midpoint:]]
                for idx, spktrain in enumerate(spktrains):
                    patterns = netools.detect_cell_assemblies(spktrain)
                    ne = NE(self.exp, self.depth, self.probe, df, stim=stim+str(idx), spktrain=spktrain,
                            patterns=patterns, edges=edges[idx])
                    ne_split[stim+str(idx)] = ne
            ne_split['dmr_first'] = self.dmr_first

        elif not hasattr(self, 'spktrain_spon'):
            print('Spontaneous activities not recorded')
        else:
            print('Sensory-evoked activities not recorded')
        return ne_split

    def get_unit_position(self):
        probdata = self.probdata
        depth = self.depth
        for unit in self.units:
            chan = unit.chan
            idx_chan = np.where(probdata['chanMap'] == chan)[0][0]
            unit.position = [int(probdata['xcoords'][idx_chan]), int(depth - probdata['ycoords'][idx_chan])]

    def get_strf(self, stim, nlead=20, nlag=0):
        """
        calculate strf for units in the session

        :param stim: array containing the spectrogram of the stimulus played during the experiment
        :param nlead: number of time binned before spikes
        :param nlag: number of time binned after spikes
        :return: None

        Note:
            The time resolution of the stimulus is the same as spktrain in the session -- 0.5ms
            To achieve a desired time resolution, down sample stimulus using .downsample method before calling the
            the session.get_strf method
            e.g. stim.downsample(df=10) to achieve a 5ms temporal resolution
        """
        edges = self.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.units:
            unit.get_strf(stim, edges, nlag=nlag, nlead=nlead)

    def get_crh(self, stim):
        edges = self.edges_dmr
        for unit in self.units:
            unit.get_crh(stim, edges)

    def get_strf_ri(self, stim, nlead=20, nlag=0, method='block', n_block=10, n_sample=1000):
        edges = self.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.units:
            unit.get_strf_ri(stim, edges, nlead=20, nlag=0, method=method,  n_block=n_block, n_sample=n_sample)

    def get_crh_ri(self, stim, method='block', n_block=10, n_sample=1000):
        edges = self.edges_dmr
        for unit in self.units:
            unit.get_crh_ri(stim, edges, method=method, n_block=n_block, n_sample=n_sample)

    def get_cluster_span(self, members, direction='vert'):
        position = []
        for member in members:
            if direction == 'vert':
                position.append(self.units[member].position[1])
            elif direction == 'horz':
                position.append(self.units[member].position[0])
        return np.max(position) - np.min(position)


class NE(Session):
    def __init__(self, exp, depth, probe, df, stim=None, spktrain=None, patterns=None, edges=None):
        self.exp = exp
        self.depth = depth
        self.probe = probe
        self.df = df
        self.stim = stim
        self.spktrain = spktrain
        self.edges = edges
        self.patterns = patterns

    def get_members(self):
        n_neuron = self.spktrain.shape[0]
        thresh = 1 / np.sqrt(n_neuron)
        members = {}
        patterns = self.patterns
        for idx, pattern in enumerate(patterns):
            members[idx] = np.where(pattern > thresh)[0]
            if len(members[idx]) < 2:
                del members[idx]
        self.ne_members = members

    def get_activity(self, member_only=True):

        patterns = self.patterns
        spktrain = self.spktrain

        if member_only:
            members = self.ne_members
        else:
            members = np.zeros(patterns.shape[0])

        activity = {}  # key is cNE index, value is numpy array of activity values
        for idx, pattern in enumerate(patterns):
            if member_only and idx not in members:
                continue
            else:
                activity[idx] = netools.get_activity(spktrain, pattern, member_only, members[idx])
        self.ne_activity = activity

    def get_activity_thresh(self, niter=20, random_state=0, member_only=True):
        """get threshold for significant activities with circular shift"""
        random.seed(random_state)
        alpha = np.arange(99, 100, 0.1)
        spktrain = np.copy(self.spktrain)
        patterns = self.patterns

        if member_only:
            members = self.ne_members
        else:
            members = np.zeros(patterns.shape[0])

        # get shuffled activity values
        activities = {}
        for i in range(patterns.shape[0]):
            activities[i] = []
        for _ in range(niter):
            # shuffle spktrain
            for i in range(spktrain.shape[0]):
                shift = random.randint(1, spktrain.shape[1])
                spktrain[i] = np.roll(spktrain[i], shift)
            # get activity for all cNEs
            for i in range(patterns.shape[0]):
                activity = netools.get_activity(spktrain, patterns[i], member_only, members[i])
                activities[i].append(activity)

        # set percentile of null activity values as threshold
        thresh = {}
        for i, activity in activities.items():
            activity = np.array(activity).flatten()
            if activity.size > 0:
                thresh[i] = np.zeros(alpha.shape)
                for idx_alpha, a in enumerate(alpha):
                    thresh[i][idx_alpha] = np.percentile(activity, a)

        self.activity_alpha = alpha
        self.activity_thresh = thresh

    def get_ne_spikes(self, alpha=99.5):

        # get session file from which the cNE is calculated
        file_path = self.file_path
        session_file_path = re.sub('-ne-.*.pkl', r'.pkl', file_path)
        with open(session_file_path, 'rb') as f:
            session = pickle.load(f)

        spiketimes = []
        for unit in session.units:
            spiketimes.append(unit.spiketimes)

        # get ne spikes for each cne and member neurons
        ne_units = []
        member_ne_spikes = {}
        idx_alpha = np.where(self.activity_alpha > alpha - .01)[0][0]
        for idx, members in self.ne_members.items():
            spiketimes_member = [spiketimes[x] for x in members]
            ne_spikes, ne_spikes_member = \
                netools.get_ne_spikes(activity=self.ne_activity[idx], thresh=self.activity_thresh[idx][idx_alpha],
                                      spiketimes=spiketimes_member, edges=self.edges)
            ne_units.append(SingleUnit(unit=idx, spiketimes=ne_spikes))
            member_ne_spikes[idx] = []
            for i, member in enumerate(members):
                member_ne_spikes[idx].append(SingleUnit(unit=member, spiketimes=ne_spikes_member[i]))

        self.ne_units = ne_units
        self.member_ne_spikes = member_ne_spikes

    def get_strf(self, stim, nlead=20, nlag=0):

        session_file_path = re.sub('-ne-.*.pkl', r'.pkl', self.file_path)
        with open(session_file_path, 'rb') as f:
            session = pickle.load(f)

        edges = session.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.ne_units:
            unit.get_strf(stim, edges, nlag=nlag, nlead=nlead)

        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf(stim, edges, nlag=nlag, nlead=nlead)

    def get_sham_patterns(self, nshift=1000):
        sham_patterns = []
        n_neuron, nt = self.spktrain.shape
        num_ne = self.patterns.shape[0]
        for shift in range(nshift):
            shift_size = np.random.randint(low=1, high=nt-1, size=n_neuron)
            spktrain_z = zscore(self.spktrain, axis=1)
            for i in range(n_neuron):
                spktrain_z[i] = np.roll(spktrain_z[i], shift_size[i])

            sham_patterns.extend(netools.fast_ica(spktrain_z, num_ne, niter=500))
        sham_patterns = np.array(sham_patterns)
        self.patterns_sham = sham_patterns

    def get_member_pairs(self):
        n_neuron = self.patterns.shape[1]
        all_pairs = set(combinations(range(n_neuron), 2))
        member_pairs = set()
        for members in self.ne_members.values():
            member_pairs.update(set(combinations(members, 2)))
        nonmember_pairs = all_pairs.difference(member_pairs)
        return member_pairs, nonmember_pairs


def save_su_df(datafolder):
    pass
