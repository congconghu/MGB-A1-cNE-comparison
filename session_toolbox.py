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
        strf = []
        if edges.ndim == 1:
            edges = [edges]
        for edge in edges:
            spktrain, _ = np.histogram(spiketimes, edge)
            stim_mat = stim.stim_mat
            taxis = (stim.taxis[1] - stim.taxis[0]) * np.array(range(nlead-1, -nlag-1, -1)) * 1000
            faxis = stim.faxis
            strf.append(netools.calc_strf(stim_mat, spktrain, nlag=nlag, nlead=nlead))
        strf = np.array(strf)
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
    
    def get_up_down_spikes(self, up_interval):
        spiketimes = self.spiketimes
        spiketimes_up = np.zeros(spiketimes.shape)
        
        idx = 0
        for start, end in zip(*up_interval):
            
            while idx < len(spiketimes) and spiketimes[idx] < end:
                if spiketimes[idx] > start:
                    spiketimes_up[idx] = 1
                
                idx += 1
        
        self.spiketimes_down = spiketimes[spiketimes_up == 0]
        self.spiketimes_up = spiketimes[spiketimes_up == 1]
    
    def get_spktrain(self, binsize=0.5, edges=None):
        spiketimes = self.spiketimes
        if edges is not None:
            spktrain, _ = np.histogram(spiketimes, bins=edges)
        else:
            spktrain, _ = np.histogram(spiketimes, bins=np.arange(0, spiketimes[-1]+binsize, binsize))
        return spktrain
    
    def get_waveform_tpd(self):
        idx = np.where(self.adjacent_chan == self.chan)
        waveform = self.waveforms_mean[idx].flatten()
        idx_trough = np.argmin(waveform)
        idx_peak = np.argmax(waveform[idx_trough:])
        self.waveform_tpd = idx_peak * .05
    
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
        if 'stimfile' in data_dict['spk']:
            stimfile = data_dict['spk']['stimfile']
            stimfile = re.findall('thalamus/(.*)', stimfile)[0]
            self.stimfile = stimfile
        for unit in data_dict['spk']['spk']:
            su = SingleUnit(unit=int(unit['unit']), spiketimes=unit['spiketimes'])
            su.add_properties(chan=int(unit['chan']), isolation=float(unit['isolation']), peak_snr=unit['peak_snr'],
                              isi_bin=float(unit['isibin']), isi_vio=float(unit['isi_vio']), template=unit['template'],
                              waveforms=unit['waveforms'], waveforms_mean=unit['waveforms_mean'],
                              waveforms_std=unit['waveforms_std'], amps=unit['amps'],
                              adjacent_chan=unit['adjacent_chan'])
            self.units.append(su)

    def save_pkl_file(self, savefile_path=None):
        if savefile_path is None:
            savefile_path = self.file_path
        else:
            self.file_path = savefile_path
        with open(savefile_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_spktrain_from_stim(self, stim_len):
        trigger = self.trigger
        trigger_ms = trigger / self.fs * 1e3
        nt = int(stim_len / max(trigger_ms.shape))  # number of time points between 2 triggers
        if len(trigger.shape) == 2: 
            n_trigger = trigger.shape[0]
        else:
            n_trigger =  1
            trigger = trigger.reshape(1, -1)
            trigger_ms = trigger_ms.reshape(1, -1)
        
        edges = []
        spktrain = []
        for i in range(n_trigger):
            trigger_ms_tmp = np.append(trigger_ms[i], [trigger_ms[i][-1] + np.max(np.diff(trigger_ms))])
            edges_tmp = np.array(
                [np.linspace(trigger_ms_tmp[x], trigger_ms_tmp[x + 1], nt + 1)[:-1] for x in range(len(trigger_ms_tmp) - 1)])
            edges_tmp = edges_tmp.flatten()
            edges_tmp = np.append(edges_tmp, trigger_ms_tmp[-1])
            edges.append(edges_tmp)

            # get spktrain under dmr
            spktrain_tmp = np.zeros([len(self.units), stim_len], 'int8')
            for idx, unit in enumerate(self.units):
                spktrain_tmp[idx], _ = np.histogram(unit.spiketimes, edges_tmp)
                spktrain_tmp[spktrain_tmp > 0] = 1
                spiketimes_dmr = unit.spiketimes[(unit.spiketimes >= edges_tmp[0]) & (unit.spiketimes < edges_tmp[-1])]
                if hasattr(self, 'spiketimes_dmr'):
                    unit.spiketimes_dmr = np.concatenate(unit.spiketimes_dmr, spiketimes_dmr, axis=0)
                else:
                    unit.spiketimes_dmr = spiketimes_dmr
                
            spktrain.append(spktrain_tmp)
        
        self.spktrain_dmr = np.array(spktrain)
        self.edges_dmr = np.array(edges)

        # get spktrain under spon
        if n_trigger == 1:
            if trigger_ms[0][0] / 1e3 / 60 > 5:  # first trigger happens 5minutes after the start of recording -> spon first
                self.dmr_first = False
                edges = [np.arange(0, trigger_ms[0][0], 0.5)]
            else:
                self.dmr_first = True
                spiketimes_max = max([unit.spiketimes[-1] for unit in self.units])
                edges = [np.arange(trigger_ms[0][-1], spiketimes_max + 0.5, 0.5)]
        else:
            edges = []
            # first spon part exist before the first trigger
            edges.append(np.arange(0, trigger_ms[0][0], 0.5))
            # second spon part exist between 2nd and 3rd trigger
            edges.append(np.arange(trigger_ms[1][-1] + 1e3, trigger_ms[2][0], 0.5))

        if len(edges[0]) * 0.5 / 1e3 / 60 > 5:  # save spon train when recording of spontanous activity lasts more than 5 minutes
            spktrain_all = []    
            for curr_edges in edges:
                spktrain = np.zeros([len(self.units), len(curr_edges) - 1], 'int8')
                for idx, unit in enumerate(self.units):
                    spktrain[idx], _ = np.histogram(unit.spiketimes, curr_edges)
                    spktrain[spktrain > 0] = 1
                    spiketimes_spon = unit.spiketimes[(unit.spiketimes >= curr_edges[0]) & (unit.spiketimes < curr_edges[-1])]
                    if hasattr(self, 'spiketimes_spon'):
                        unit.spiketimes_spon = np.concatenate(unit.spiketimes_spon, spiketimes_spon, axis=0)
                    else:
                        unit.spiketimes_spon = spiketimes_spon
                spktrain_all.append(spktrain)
               
            self.spktrain_spon = spktrain_all
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
        
        
        if type(spktrain) == np.array and spktrain.ndim == 2:
            spktrain = np.array([spktrain])
            edges = np.array([edges])
        # down sample spktrain
        spktrain_downsampled = []
        for spktrain_tmp in spktrain:
            nt = spktrain_tmp.shape[-1] // df
            spktrain_tmp = spktrain_tmp[:, :nt * df]
            spktrain_tmp = np.resize(spktrain_tmp, (spktrain_tmp.shape[0], nt, df))
            spktrain_downsampled.append(np.sum(spktrain_tmp, axis=2))
        spktrain = np.concatenate(spktrain_downsampled, axis=1)
        # down sample edges
        edges_downsampled = []
        for edge in edges:
            nt = edge.shape[-1] // df
            edges_downsampled.append(edge[0:(nt * df + 1):df])
        edges = edges_downsampled
        return spktrain, edges

    def get_ne(self, df, stim, shuffle=False):
        """get cNEs in each recording session, under dmr and spon"""
        # get cNE for dmr-evoked activity
        spktrain, edges = self.downsample_spktrain(df, stim)
        if shuffle:
            spktrain = netools.shuffle_spktrain(spktrain)
        if spktrain.any():
            patterns = netools.detect_cell_assemblies(spktrain)
            if patterns is None:
                return None
            else:
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
            edges_df = []
            for edge in edges:
                edges_df.append(edge[::df])
            edges = np.array(edges_df)
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
    
    def get_up_down_data(self, datafolder=None):
        if datafolder is None:
            file_path = self.file_path
            datafolder = re.findall('(.*)\d{6}_\d{6}', file_path)[0]
        exp = self.exp
        depth = self.depth
        up_down_file = glob.glob(os.path.join(datafolder, '{}-*-{}um-*up_down.pkl'.format(exp, depth)))
        assert(len(up_down_file) == 1)
        with open(up_down_file[0], 'rb') as f:
            up_down = pickle.load(f)
        return up_down
    
    def get_up_down_spikes(self, datafolder=None, stim='spon'):
        up_down = self.get_up_down_data(datafolder)
        up_interval = up_down['up_interval_'+stim]
        for unit in self.units:
            unit.get_up_down_spikes(up_interval)
    
    def save_spktrain_up_state(self, stim='spon'):
        up_down = self.get_up_down_data()
        up_intervals = up_down['up_interval_'+stim]
        spktrain = eval(f'self.spktrain_{stim}')
        edges = eval(f'self.edges_{stim}')
        spktrain_up = []
        edges_up = []
        p = 0
        for t_start, t_end in zip(*up_intervals):
            for idx_start in range(p, len(edges)):
                if edges[idx_start] >= t_start:
                    for idx_end in range(idx_start, len(edges)):
                        if edges[idx_end] >= t_end:
                            p = idx_end
                            spktrain_up.append(spktrain[:, idx_start:idx_end])
                            edges_up.append(edges[idx_start:idx_end])
                            break
                    break
        edges_up.append([edges[-1]])
        spktrain_up = np.concatenate(spktrain_up, axis=1)
        edges_up = np.concatenate(edges_up)
        if stim == 'spon':
            self.spktrain_spon_up = spktrain_up
            self.edges_spon_up = edges_up
        else:
            self.spktrain_dmr_up = spktrain_up
            self.edges_dmr_up = edges_up
        self.save_pkl_file()


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
            spktrain = netools.shuffle_spktrain(spktrain)
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
            if isinstance(edges, list) or edges.ndim > 1:
                edges_tmp = []
                for edge in edges:
                    edge = edge.flatten()
                    edges_tmp.append(edge[::df])
                edges = np.array(edges_tmp)
            else:
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
            isi = np.diff(ne_unit.spiketimes, prepend=0)
            ne_unit.spiketimes = ne_unit.spiketimes[isi > 10]
            if ne_unit.spiketimes.shape[0] < nspk_thresh:
                continue
            
            for m, member in enumerate(members):
                ne_spike_unit = self.member_ne_spikes[i][m]
                isi = np.diff(ne_spike_unit.spiketimes, prepend=0)
                ne_spike_unit.spiketimes = ne_spike_unit.spiketimes[isi > 10]
                if ne_spike_unit.spiketimes.shape[0] < nspk_thresh:
                    continue
                unit = session.units[member]
                isi = np.diff(unit.spiketimes, prepend=0)
                unit.spiketimes = unit.spiketimes[isi > 10]
                unit.spiketimes = unit.spiketimes[(unit.spiketimes > edges[0]) & (unit.spiketimes < edges[-1])]
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
                mi_raw = np.empty((3, nsample))
                
                for rf in ('strf', 'crh'):
                    rf_strf_crh[rf] = np.empty((3, *eval('unit.{}.shape'.format(rf)), nsample))
                    ri_strf_crh[rf]  = np.empty((3, 1000, nsample))
                    
                for n in range(nsample):
                    
                    unit_tmp = deepcopy(unit)
                    ne_unit_tmp = deepcopy(ne_unit)
                    ne_spike_unit_tmp = deepcopy(ne_spike_unit)
                
                    for u, curr_unit in enumerate((unit_tmp, ne_unit_tmp, ne_spike_unit_tmp)):
                        curr_unit.subsample(n_events)
                        
                        for rf in ('strf', 'crh'):
                            eval('curr_unit.get_{rf}(stim_{rf}, edges_{rf})'.format(rf=rf))
                            eval('curr_unit.get_{rf}_ri(stim_{rf}, edges_{rf}, return_null=False)'.format(rf=rf))
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
                        mi_raw[u, n] = curr_unit.strf_ifrac[:, :, -1].mean()
                
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
                
                for param in ('ptd', 'morani', 'asi', 'mi','mi_raw', 'nonlin_centers', 'nonlin_fr'):
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
    
    def get_up_down_data(self, datafolder):
        exp = re.findall('\d{6}_\d{6}', self.file_path)[0]
        depth =  re.findall('\d{3,4}um', self.file_path)[0]
        up_down_file = glob.glob(os.path.join(datafolder, '{}-*-{}-*up_down.pkl'.format(exp, depth)))
        if len(up_down_file) == 0:
            return None
        assert(len(up_down_file) == 1)
        with open(up_down_file[0], 'rb') as f:
            up_down = pickle.load(f)
        return up_down
    
    def get_up_down_spikes(self, datafolder, stim='spon'):
        up_down = self.get_up_down_data(datafolder)
        if up_down is None:
            return False
        up_interval = up_down['up_interval_'+stim]
        # cNE events
        for unit in self.ne_units:
            unit.get_up_down_spikes(up_interval)

        # get member ne spike 
        for _, members in self.member_ne_spikes.items():
            for unit in members:
                unit.get_up_down_spikes(up_interval)
        return True


def save_su_df(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    """
    save single unit data into DataFrame
    :param datafolder:
    :param savefolder:
    :return:
    """
    droplist = ['spiketimes', 'isi_bin', 'waveforms', 'waveforms_mean', 'waveforms_std', 'amps',
                'adjacent_chan', 'template', 'peak_snr', 'strf_nonlinearity', 'si_null', 'si_spk', 
                'nonlin_t_bins', 'nonlin_nspk_bins', 'strf_ifrac', 'strf_info_xcenters']
    files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
    units_all = []
    for idx, file in enumerate(files):
        print('({}/{}) save dataframe for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            session = pickle.load(f)

        units = pd.DataFrame([vars(x) for x in session.units])
        
        for colname in units.columns:
            if colname in droplist:
                units.drop(colname, axis=1, inplace=True)
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
        

def save_matched_ne_df(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\comparison\data-summary', key='dmr'):
    
    files = glob.glob(datafolder + r'\*ne-20dft-{}.pkl'.format(key), recursive=False)
    ne_all = []
    for idx, file in enumerate(files):
        print('({}/{}) save dataframe for {}'.format(idx + 1, len(files), file))
        if key == 'dmr':
            file2 = re.sub('20dft-dmr', '20dft-spon', file)
        elif key == 'dmr_up':
            file2 = re.sub('20dft-dmr_up', '20dft-dmr', file)
        if not os.path.exists(file2):
            print('{} does not exist'.format(file2))
            continue
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        with open(file2, 'rb') as f:
            ne2 = pickle.load(f)
            
        exp = ne.exp
        probe = ne.probe
        depth = ne.depth
        for i, members in ne.ne_members.items():
            pattern1 = ne.patterns[i]
            if hasattr(ne, 'ne_units'):
                strf_sig = ne.ne_units[i].strf_ri_z > 3
                crh_sig = ne.ne_units[i].crh_ri_z > 3
            else:
                strf_sig = None
                crh_sig = None
            if i in ne.pattern_order[0]:
                idx_match = np.where(np.array(ne.pattern_order[0]) == i)[0][0]
                idx2 = ne.pattern_order[1][idx_match]
                pattern2 = ne2.patterns[idx2]
                pattern_corr = ne.corrmat[idx_match, idx_match]
                
            else:
                pattern2 = []
                pattern_corr = np.nan
            if hasattr(ne, 'corr_thresh'):
                corr_thresh = ne.corr_thresh
            else:
                corr_thresh = None
            ne_all.append({'exp': exp, 'probe': probe, 'depth':depth,
                           'cNE': i, 'members': members, 
                           'pattern_dmr': pattern1, 'pattern_spon': pattern2, 
                           'pattern_corr': pattern_corr, 'corr_thresh': corr_thresh,
                           'strf_sig': strf_sig, 'crh_sig': crh_sig})
    ne_all = pd.DataFrame(ne_all)
    ne_all.to_json(os.path.join(savefolder, 'cNE_matched-{}.json'.format(key)))
    
def save_matched_shuffled_ne_df(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    
    files = glob.glob(datafolder + r'\*ne-20dft-dmr-shuffled.pkl', recursive=False)
    ne_all = []
    for idx, file in enumerate(files):
        print('({}/{}) save dataframe for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            ne_list = pickle.load(f)
        
        if not 'corrmat' in ne_list: continue
        c = 0
        for ne in ne_list['ne']:
            if ne is None:
                continue
            exp = ne.exp
            probe = ne.probe
            depth = ne.depth
            break
        
        for corrmat in ne_list['corrmat']:
            n_ne = corrmat.shape[0]
            for i in range(n_ne):
                ne_all.append({'exp': exp, 'probe': probe, 'depth':depth,'cNE': c,  
                           'pattern_corr': corrmat[i, i]})
                c += 1
    ne_all = pd.DataFrame(ne_all)
    ne_all.to_json(os.path.join(savefolder, 'cNE_matched_shuffled.json'))
   

def ne_neuron_subsample(stim_strf=None, stim_crh=None, 
                           datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
                           savefolder=r'E:\Congcong\Documents\data\comparison\data-summary'):
    
    subsample = []
    
    files = glob.glob(os.path.join(datafolder, '*-20dft-dmr.pkl'))
    for idx, file in enumerate(files):
        print('({}/{}) getting subsampled ri for {}'.format(idx + 1, len(files), file))
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        exp = ne.exp
        probe = ne.probe
        subsample = ne.get_subsampled_properties(stim_strf=stim_strf, stim_crh=stim_crh, nspk_thresh=100)
        subsample.reset_index(drop=True, inplace=True)
        subsample.to_json(os.path.join(savefolder, '{}-{}-subsample_ri.json'.format(exp, probe)))
    
    
def ne_neuron_subsample_combine(datafolder='E:\Congcong\Documents\data\comparison\data-summary\subsample'):
    ri_file = glob.glob(os.path.join(datafolder, '*subsample_ri.json'))
    subsample_ri = []
    for file in ri_file:
        subsample_ri.append(pd.read_json(file))
    subsample_ri = pd.concat(subsample_ri)
    subsample_ri.reset_index(drop=True, inplace=True)
    subsample_ri.to_json('E:\Congcong\Documents\data\comparison\data-summary\subsample_ne_neuron.json')
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


        