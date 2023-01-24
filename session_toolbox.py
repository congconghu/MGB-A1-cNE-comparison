import glob
import os
import pickle
import re
import random
import math

import mat73
import numpy as np
import pandas as pd

from scipy.stats import zscore
from scipy.io import loadmat
import ne_toolbox as netools
from itertools import combinations
from copy import deepcopy

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

    def get_strf_ri(self, stim, edges, nlead=20, nlag=0, method='block', n_block=10, n_sample=1000, return_null=True):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)
        stim_mat = stim.stim_mat

        ri = netools.calc_strf_ri(spktrain, stim_mat, nlead=nlead, nlag=nlag, method=method,
                                      n_block=n_block, n_sample=n_sample, bigmat_file=stim.bigmat_file)
        self.strf_ri = ri
        
        if return_null:
            spktrain_null = spktrain[::-1]
            ri_null = netools.calc_strf_ri(spktrain_null, stim_mat, nlead=nlead, nlag=nlag, method=method,
                                           n_block=n_block, n_sample=n_sample, bigmat_file=stim.bigmat_file)
            self.strf_ri_null = ri_null
            self.strf_ri_z = (ri.mean() - ri_null.mean()) / np.sqrt((ri.std() ** 2 + ri_null.std() ** 2) / 2)
            self.strf_ri_p = sum(ri_null > ri.mean()) / n_sample

    def get_crh_ri(self, stim, edges, method='block', n_block=10, n_sample=1000, return_null=True):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)

        ri = netools.calc_crh_ri(spktrain, stim, method=method, n_block=n_block, n_sample=n_sample)
        self.crh_ri = ri
        
        if return_null:
            spktrain_null = spktrain[::-1]
            ri_null = netools.calc_crh_ri(spktrain_null, stim, method=method, n_block=n_block, n_sample=n_sample)
        
            self.crh_ri_null = ri_null
            self.crh_ri_z = (ri.mean() - ri_null.mean()) / np.sqrt((ri.std() ** 2 + ri_null.std() ** 2) / 2)
            self.crh_ri_p = sum(ri_null > ri.mean()) / n_sample

    def get_strf_properties(self):
        bf, latency = netools.calc_strf_properties(self.strf, self.strf_taxis, self.strf_faxis)
        self.bf = bf
        self.latency = latency

    def get_strf_significance(self, criterion='z', thresh=3):
        if criterion == 'z':
            self.strf_sig = self.strf_ri_z > thresh
    
    def get_strf_ptd(self):
        self.strf_ptd = netools.calc_strf_ptd(self.strf, len(self.spiketimes))
        
    def get_strf_nonlinearity(self, edges, stim):
        spiketimes = self.spiketimes
        spktrain, _ = np.histogram(spiketimes, edges)
        strf_nonlinearity = netools.calc_strf_nonlinearity(self.strf, spktrain, stim)
        
        self.si_null =  strf_nonlinearity['si_null']
        self.si_spk =  strf_nonlinearity['si_spk']
        self.nonlin_centers =  strf_nonlinearity['centers']
        self.nonlin_t_bins =  strf_nonlinearity['t_bins']
        self.nonlin_nspk_bins =  strf_nonlinearity['nspk_bins']
        self.nonlin_fr =  strf_nonlinearity['fr']
        self.nonlin_fr_mean =  strf_nonlinearity['fr_mean']
        self.nonlin_asi =  strf_nonlinearity['asi']
        
    def get_strf_mi(self, edges, stim):
            spiketimes = self.spiketimes
            self.strf_info, self.strf_ifrac, self.strf_info_xcenters = netools.calc_strf_mi(spiketimes, edges, stim)
            
    def get_crh_properties(self):
        btmf, bsmf = netools.calc_crh_properties(self.crh, self.tmfaxis, self.smfaxis)
        self.btmf = btmf
        self.bsmf = bsmf
    
    def get_crh_morani(self):
        self.crh_morani = netools.moran_i(self.crh)

    def get_crh_significance(self, criterion='z', thresh=3):
        if criterion == 'z':
            self.crh_sig = self.crh_ri_z > thresh
    
    def subsample(self, nspk):
        self.spiketimes = np.array(random.sample(set(self.spiketimes), int(nspk)))


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
    
    def get_strf_properties(self):
        for unit in self.units:
            unit.get_strf_properties()
            
    def get_strf_ptd(self):
        for unit in self.units:
            unit.get_strf_ptd()
    
    def get_strf_nonlinearity(self, stim):
        edges = self.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.units:
            unit.get_strf_nonlinearity(edges, stim.stim_mat)
    
    def get_strf_mi(self, stim):
        edges = self.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.units:
            unit.get_strf_mi(edges, stim.stim_mat)

    def get_strf_ri(self, stim, nlead=20, nlag=0, method='block', n_block=10, n_sample=1000):
        edges = self.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.units:
            unit.get_strf_ri(stim, edges, nlead=20, nlag=0, method=method,  n_block=n_block, n_sample=n_sample)
    
    def get_strf_significance(self, criterion='z', thresh=3):
        for unit in self.units:
            unit.get_strf_significance(criterion=criterion, thresh=thresh)    
    
    def get_crh(self, stim):
        edges = self.edges_dmr
        for unit in self.units:
            unit.get_crh(stim, edges)

    def get_crh_properties(self):
        for unit in self.units:
            unit.get_crh_properties()
    
    def get_crh_morani(self):
        for unit in self.units:
            unit.get_crh_morani()

    def get_crh_ri(self, stim, method='block', n_block=10, n_sample=1000):
        edges = self.edges_dmr
        for unit in self.units:
            unit.get_crh_ri(stim, edges, method=method, n_block=n_block, n_sample=n_sample)
            
    def get_crh_significance(self, criterion='z', thresh=3):
        for unit in self.units:
            unit.get_crh_significance(criterion=criterion, thresh=thresh)

    def get_cluster_span(self, members, direction='vert'):
        position = []
        for member in members:
            if direction == 'vert':
                position.append(self.units[member].position[1])
            elif direction == 'horz':
                position.append(self.units[member].position[0])
        return np.max(position) - np.min(position)

    def get_cluster_freq_span(self, members):
        freq = []
        for member in members:
            if self.units[member].strf_sig:
                freq.append(self.units[member].bf)
        if len(freq) >= 2:
            return math.log2(np.max(freq) / np.min(freq))
        else:
            return np.nan


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
        session = self.get_session_data()

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

        session = self.get_session_data()
        edges = session.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        for unit in self.ne_units:
            unit.get_strf(stim, edges, nlag=nlag, nlead=nlead)

        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf(stim, edges, nlag=nlag, nlead=nlead)
    
    def get_strf_ri(self, stim, nlead=20, nlag=0, method='block', n_block=10, n_sample=1000):
        
        session = self.get_session_data()
        edges = session.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
        
        # get strf ri for cNE events
        for unit in self.ne_units:
            unit.get_strf_ri(stim, edges, nlead=20, nlag=0, method=method,  n_block=n_block, n_sample=n_sample)

        # get member strf ri
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf_ri(stim, edges, nlead=20, nlag=0, method=method,  n_block=n_block, n_sample=n_sample)
                
    def get_strf_properties(self):
        
        # get strf ri for cNE events
        for unit in self.ne_units:
            unit.get_strf_properties()

        # get member strf ri
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf_properties()
                
    def get_strf_significance(self, criterion='z', thresh=3):
        # get strf ri for cNE events
        for unit in self.ne_units:
            unit.get_strf_significance(criterion=criterion, thresh=thresh)

        # get member strf ri
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf_significance(criterion=criterion, thresh=thresh)
    
    def get_strf_ptd(self):
        # get strf ptd for cNE events
        for unit in self.ne_units:
            unit.get_strf_ptd()

        # get member ne spike strf ptd
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf_ptd()
                
    def get_strf_nonlinearity(self, stim):
        session = self.get_session_data()
        edges = session.edges_dmr
        if hasattr(stim, 'df'):
            df = stim.df
            edges = edges[::df]
            
        # get strf ptd for cNE events
        for unit in self.ne_units:
            unit.get_strf_nonlinearity(edges, stim.stim_mat)

        # get member ne spike strf ptd
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_strf_nonlinearity(edges, stim.stim_mat)
                
    def get_strf_info(self, stim):
         session = self.get_session_data()
         edges = session.edges_dmr
         if hasattr(stim, 'df'):
             df = stim.df
             edges = edges[::df]
             
         # get strf ptd for cNE events
         for unit in self.ne_units:
             unit.get_strf_mi(edges, stim.stim_mat)

         # get member ne spike strf ptd
         for _, members in self.member_ne_spikes.items():
             for unit in members:
                 unit.get_strf_mi(edges, stim.stim_mat)
        
    def get_crh(self, stim):
        
        session = self.get_session_data()
        edges = session.edges_dmr
        
        for unit in self.ne_units:
            unit.get_crh(stim, edges)

        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_crh(stim, edges)
                
    def get_crh_ri(self, stim, method='block', n_block=10, n_sample=1000):
       
         session = self.get_session_data()
         edges = session.edges_dmr
         
         # get strf ri for cNE events
         for unit in self.ne_units:
             unit.get_crh_ri(stim, edges, method=method, n_block=n_block, n_sample=n_sample)

         # get member strf ri
         for _, members in self.member_ne_spikes.items():
             for unit in members:
                 unit.get_crh_ri(stim, edges, method=method, n_block=n_block, n_sample=n_sample)
                 
    def get_crh_properties(self):
         
         # get strf ri for cNE events
         for unit in self.ne_units:
             unit.get_crh_properties()

         # get member strf ri
         for _, members in self.member_ne_spikes.items():
             for unit in members:
                 unit.get_crh_properties()
                 
    def get_crh_significance(self, criterion='z', thresh=3):
         # get crh significance for cNE events
         for unit in self.ne_units:
             unit.get_crh_significance(criterion=criterion, thresh=thresh)

         # get member crh sig
         for _, members in self.member_ne_spikes.items():
             for unit in members:
                 unit.get_crh_significance(criterion=criterion, thresh=thresh)
    
    def get_crh_morani(self):
        # get strf ptd for cNE events
        for unit in self.ne_units:
            unit.get_crh_morani()

        # get member ne spike strf ptd
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_crh_morani()
    
    def get_subsampled_properties(self, stim_strf=None, stim_crh=None, nspk_thresh=100, nsample=10):
        session = self.get_session_data()
        edges = session.edges_dmr
        if hasattr(stim_strf, 'df'):
            df = stim_strf.df
            edges_strf = edges[::df]
        edges_crh = edges
            
        ri_all = []
        for i, members in self.ne_members.items():
            print('ne #{}/{}'.format(i, len(self.ne_members)))
            ne_unit = self.ne_units[i]
            if ne_unit.spiketimes.shape[0] < nspk_thresh:
                continue
            
            for m, member in enumerate(members):
                ne_spike_unit = self.member_ne_spikes[i][m]
                if ne_spike_unit.spiketimes.shape[0] < nspk_thresh:
                    continue
                unit = session.units[member]
                min_events = min(ne_unit.spiketimes.shape[0], ne_spike_unit.spiketimes.shape[0])
                n_events = np.floor(min_events * 0.9)
                # repeat 10 times for subsampling
                rf_strf_crh = {'strf':[], 'crh':[]}
                ri_strf_crh =  {'strf':[], 'crh':[]}
                ptd = np.empty((3, nsample))
                morani = np.empty((3, nsample))
                asi = np.empty((3, nsample))
                nonlin_centers  = [[], [], []]
                nonlin_fr = [[], [], []]
                mi = np.empty((3, nsample))
                
                for rf in ('strf', 'crh'):
                    rf_strf_crh[rf] = np.empty((3, *eval('unit.{}.shape'.format(rf)), nsample))
                    ri_strf_crh[rf]  = np.empty((3, 1000, nsample))
                    
                for n in range(nsample):
                    
                    unit_tmp = deepcopy(unit)
                    ne_unit_tmp = deepcopy(ne_unit)
                    ne_spike_unit_tmp = deepcopy(ne_spike_unit)
                
                    for u, curr_unit in enumerate((unit_tmp, ne_unit_tmp, ne_spike_unit_tmp)):
                        random.seed(n)
                        curr_unit.subsample(n_events)
                        
                        for rf in ('strf', 'crh'):
                            eval('curr_unit.get_{}(stim_{}, edges_{})'.format(rf, rf, rf))
                            eval('curr_unit.get_{}_ri(stim_{}, edges_{}, return_null=False)'.format(rf, rf, rf))
                            rf_strf_crh[rf][u, :, :, n] = eval('curr_unit.{}'.format(rf))
                            ri_strf_crh[rf][u, :, n] = eval('curr_unit.{}_ri'.format(rf))
                        
                        curr_unit.get_strf_ptd()
                        curr_unit.get_crh_morani()
                        curr_unit.get_strf_nonlinearity(edges_strf, stim_strf.stim_mat)
                        curr_unit.get_strf_mi(edges_strf, stim_strf.stim_mat)
                        ptd[u, n] =  curr_unit.strf_ptd
                        morani[u, n] =  curr_unit.crh_morani
                        asi[u, n] =  curr_unit.nonlin_asi
                        nonlin_centers[u].append(curr_unit.nonlin_centers)
                        nonlin_fr[u].append(curr_unit.nonlin_fr)
                        mi[u, n] =  np.mean(curr_unit.strf_info)
                
                ri = {'exp': session.exp, 'depth': session.depth, 'probe': session.probe, 
                                'cNE': i, 'member': member, 'n_events': n_events}
                for rf in ('strf', 'crh'):
                    if eval('stim_{}'.format(rf)) is not None:
                    
                        ri.update({rf + '_neuron': rf_strf_crh[rf][0], 
                               rf + '_cNE': rf_strf_crh[rf][1], 
                               rf + '_ne_spike': rf_strf_crh[rf][2],
                               rf + '_ri_neuron': ri_strf_crh[rf][0],
                               rf + '_ri_cNE': ri_strf_crh[rf][1],
                               rf + '_ri_ne_spike': ri_strf_crh[rf][2]})
                
                for param in ('ptd', 'morani', 'asi', 'mi', 'nonlin_centers', 'nonlin_fr'):
                    ri.update({param + '_neuron': eval(param + '[0]'), 
                           param + '_cNE': eval(param + '[1]'), 
                           param + '_ne_spike': eval(param + '[2]')})
                    
                ri_all.append(ri)
        ri_all = pd.DataFrame(ri_all)
        return ri_all
                
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
    
    def get_session_data(self):
        session_file_path = re.sub('-ne-.*.pkl', r'.pkl', self.file_path)
        with open(session_file_path, 'rb') as f:
            session = pickle.load(f)
        return session


def save_su_df(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    """
    save single unit data into DataFrame
    :param datafolder:
    :param savefolder:
    :return:
    """
    files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
    units_all = []
    for idx, file in enumerate(files):
        print('({}/{}) save dataframe for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            session = pickle.load(f)

        units = pd.DataFrame([vars(x) for x in session.units])
        for i in units.index:
            units.iloc[0]['strf_nonlinearity']
        units.drop(['spiketimes', 'isi_bin', 'waveforms', 'waveforms_mean', 'waveforms_std', 'amps',
                    'adjacent_chan', 'template', 'peak_snr', 'strf_nonlinearity', 'si_null', 'si_spk', 
                    'nonlin_t_bins', 'nonlin_nspk_bins', 'strf_ifrac', 'strf_info_xcenters'], axis=1, inplace=True)
        units['exp'] = session.exp
        units['depth'] = session.depth
        units['probe'] = session.probe
        units_all.append(units)

    units = pd.concat(units_all)
    units = units.astype({'unit': 'int16', 'chan': 'int8', 'depth': 'int16'})
    units.reset_index(inplace=True)
    units.to_json(os.path.join(savefolder, 'single_units.json'))


def save_session_df(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    """
    save single unit data into DataFrame
    :param datafolder:
    :param savefolder:
    :return:
    """
    files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
    sessions = []
    for idx, file in enumerate(files):
        print('({}/{}) save dataframe for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            session = pickle.load(f)
        rem_list = ['filt_params', 'fs', 'probdata', 'units', 'trigger', 'spktrain_dmr', 'spktrain_spon',
                    'edges_dmr', 'edges_spon']
        session_vars = vars(session)
        session_vars = {key: session_vars[key] for key in session_vars.keys() if key not in rem_list}
        session_vars['n_neuron'] = len(session.units)
        session = pd.DataFrame(session_vars, index=[0])
        sessions.append(session)

    sessions = pd.concat(sessions)
    sessions = sessions.astype({'depth': 'int16', 'site': 'int8', 'n_neuron': 'int8'})
    sessions.reset_index(drop=True, inplace=True)
    sessions.to_json(os.path.join(savefolder, 'sessions.json'))


def save_ne_df(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    files = glob.glob(datafolder + r'\*ne-20dft-dmr.pkl', recursive=False)
    ne_all = []
    for idx, file in enumerate(files):
        
        print('({}/{}) save dataframe for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        exp = ne.exp
        probe = ne.probe
        depth = ne.depth
        for i, members in ne.ne_members.items():
            pattern = ne.patterns[i]
            strf_sig = ne.ne_units[i].strf_ri_z > 3
            crh_sig = ne.ne_units[i].crh_ri_z > 3
            ne_all.append({'exp': exp, 'probe': probe, 'depth':depth,
                           'cNE': i, 'members': members, 'pattern': pattern, 
                           'strf_sig': strf_sig, 'crh_sig': crh_sig})
    ne_all = pd.DataFrame(ne_all)
    ne_all.to_json(os.path.join(savefolder, 'cNE.json'))

def ne_neuron_subsample(stim_strf=None, stim_crh=None, 
                           datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
                           savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    
    subsample = []
    
    files = glob.glob(os.path.join(datafolder, '*20dft-dmr.pkl'))
    for idx, file in enumerate(files):
        print('({}/{}) getting subsampled ri for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        exp = ne.exp
        probe = ne.probe
        subsample = ne.get_subsampled_properties(stim_strf=stim_strf, stim_crh=stim_crh)
        subsample.reset_index(drop=True, inplace=True)
        subsample.to_json(os.path.join(savefolder, '{}-{}-subsample_ri.json'.format(exp, probe)))
    
def ne_neuron_subsample_combine(datafolder='E:\Congcong\Documents\data\comparison\data-summary\subsample'):
    ri_file = glob.glob(os.path.join(datafolder, '*subsample_ri.json'))
    subsample_ri = []
    for file in ri_file:
        subsample_ri.append(pd.read_json(file))
    subsample_ri = pd.concat(subsample_ri)
    subsample_ri.reset_index(drop=True, inplace=True)
    subsample_ri.to_json('E:\Congcong\Documents\data\comparison\data-summary\subsample_ri.json')
    region = ['MGB' if x == 'H31x64' else 'A1' for x in subsample_ri.probe]
    subsample_ri['region'] = region

    for rf in ('strf', 'crh'):
        for ri in ('_ri_neuron', '_ri_cNE', '_ri_ne_spike'):
            # flatten the columns contaning ri
            subsample_ri[rf+ri] = subsample_ri[rf+ri].apply(lambda x:np.array(x, dtype=np.float64)).apply(lambda x: x.flatten())
            # get ri std
            subsample_ri[rf+ri+'_std'] =  subsample_ri[rf+ri].apply(np.nanstd)
            # get ri mean
            subsample_ri[rf+ri+'_mean'] =  subsample_ri[rf+ri].apply(np.nanmean)
        
        