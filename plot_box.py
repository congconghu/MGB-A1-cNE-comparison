# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:47:51 2022

@author: Congcong
"""
import re
import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pylab import cm

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
figure_size=(18.3, 24.7)


def plot_5ms_strf_ne_and_members(ne, savepath):
    xspace = 0.1
    yspace = 0.08
    figx = 0.25
    figy = 0.2
    session_file_path = re.sub('-ne-.*.pkl', r'.pkl', ne.file_path)
    with open(session_file_path, 'rb') as f:
            session = pickle.load(f)
    n_neuron = ne.spktrain.shape[0]
    member_thresh = 1/np.sqrt(n_neuron)  # threshol for membership
    
    for cne in ne.ne_units:
        
        fig = plt.figure(figsize=figure_size)
        
        # plot cNE strf
        ax = fig.add_axes([xspace/2, figy*4+yspace/2, figx-xspace, figy-yspace])
        plot_strf(ax, cne.strf, taxis=cne.strf_taxis, faxis=cne.strf_faxis)
        
        # plot ICweight
        idx_ne = cne.unit
        weights = ne.patterns[idx_ne]
        ax = fig.add_axes([xspace/2+figx, figy*4+yspace/2, 2*figx-xspace, figy-yspace])
        plot_ICweight(ax, weights, member_thresh)
        
        # plot member strf
        members = ne.ne_members[idx_ne]
        for idx, member in enumerate(members):
            unit = session.units[member]
            nrow = 3 - idx//4
            ncol = idx % 4
            ax = fig.add_axes([xspace/2 + figx*ncol, figy*nrow+yspace/2, 
                               figx-xspace, figy-yspace])
            plot_strf(ax, unit.strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis)
            ax.set_title('unit {}'.format(unit.unit))
        name_base = re.findall('\d{6}_\d{6}.*', session_file_path)
        name_base = re.sub('.pkl', '-cNE_{}.jpg'.format(idx_ne), name_base[0])
        fig.savefig(os.path.join(savepath, name_base))
        plt.close(fig)
        

def plot_strf(ax, strf, taxis, faxis):
    
   
    max_val = abs(strf).max()*1.01
    ax.imshow(strf, aspect='auto', origin='lower', cmap='RdBu_r', 
                   vmin=-max_val, vmax=max_val)
    
    tlabels = np.array([75, 50, 25, 0])
    xticks = np.searchsorted(-taxis, -tlabels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(tlabels)
    ax.set_xlabel('time before spike (ms)')
    
    faxis = faxis/1000
    flabels = ['0.5', '2', '8', '32']
    flabels_arr = np.array([0.5, 2, 8, 32])
    yticks = np.searchsorted(faxis, flabels_arr)
    ax.set_yticks(yticks)
    ax.set_yticklabels(flabels)
    ax.set_ylabel('frequency (kHz)', labelpad=-2)


def plot_ICweight(ax, weights, thresh, direction='vertical'):
    markersize=10
    n_neuron = len(weights)
    
    # plot threshold of membership
    if direction=='vertical':
        ax.plot([0, n_neuron+1], [thresh, thresh], 'r--')
    
    # stem plot for all weights of neurons
    markerline, stemline, baseline = ax.stem(
        range(1, n_neuron+1),weights, 
        markerfmt='ok', linefmt='k-', basefmt='k-')
    plt.setp(markerline, markersize = markersize)
    # plot baseline at 0
    ax.plot([0, n_neuron+1], [0, 0], 'k-')
    
    # plot member stems
    members = np.where(weights > thresh)[0]
    markerline, _, _ = ax.stem(
        members+1, weights[members], 
        markerfmt='or', linefmt='r-', basefmt='k-')
    plt.setp(markerline, markersize = markersize)
    
    ax.set_xlim([0, n_neuron+1])
    ax.set_xticks(range(1, n_neuron+1, 5))
    ax.set_xlabel('neuron #')
    ax.set_ylabel('ICweight')

   
    