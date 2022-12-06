from scipy.io import loadmat
import os
import re
import mat73
import pickle
import numpy as np


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
        with open(savefile_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


class SingleUnit:

    def __init__(self, unit, chan, spiketimes, isolation, peak_snr, isi_bin, isi_vio,
                 waveforms=None, waveforms_mean=None, waveforms_std=None, amps=None,
                 adjacent_chan=None, template=None):
        self.unit = unit
        self.chan = chan
        self.spiketimes = spiketimes
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


class Session:

    def __init__(self, exp=None, site=None, depth=None, filt_params=None, fs=None, probe=None, probdata=None, units=[]):
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
        self.units = []
        for unit in data_dict['spk']['spk']:
            su = SingleUnit(unit=int(unit['unit']), chan=int(unit['chan']), spiketimes=unit['spiketimes'],
                            isolation=float(unit['isolation']), peak_snr=unit['peak_snr'],
                            isi_bin=float(unit['isibin']), isi_vio=float(unit['isi_vio']),
                            waveforms=unit['waveforms'], waveforms_mean=unit['waveforms_mean'],
                            waveforms_std=unit['waveforms_std'], amps=unit['amps'], adjacent_chan=unit['adjacent_chan'],
                            template=unit['template'])
            self.units.append(su)

    def save_pkl_file(self, savefile_path):
        with open(savefile_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)