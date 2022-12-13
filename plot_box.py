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
import ne_toolbox as netools

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
figure_size=(18.3, 24.7)
activity_alpha = 99.5


def plot_ne_construction(ne, savepath):
    
    # load seesion file
    session_file_path = re.sub('-ne-.*.pkl', r'.pkl', ne.file_path)
    with open(session_file_path, 'rb') as f:
            session = pickle.load(f)
    
    fig = plt.figure(figsize=figure_size)
    xspace = 0.06
    yspace = 0.08
    figx = 0.2
    figy = 0.22
    
    # plot correlation matrix
    ax = fig.add_axes([xspace, 0.75, figx-xspace, figy-yspace])
    corr_mat = np.corrcoef(ne.spktrain)
    plot_corrmat(ax, corr_mat)
    
    # plot eigenvalues
    ax = fig.add_axes([xspace+figx, 0.75, figx-xspace, figy-yspace])
    corr_mat = np.corrcoef(ne.spktrain)
    thresh = netools.get_pc_thresh(ne.spktrain)
    plot_eigen_values(ax, corr_mat, thresh)
    
    # plot IC
    ax = fig.add_axes([xspace+2*figx, 0.75, figx-xspace, figy-yspace])
    plot_ICweigh_imshow(ax, ne.patterns, ne.ne_members)
    
    # plot each IC
    n_ne, n_neuron = ne.patterns.shape
    xstart = 3*figx + xspace
    figx = (1-xstart) / n_ne
    thresh = 1/np.sqrt(n_neuron)
    ymax = ne.patterns.max()*1.1
    ymin = ne.patterns.min()*1.1
    for i in range(n_ne):
        ax = fig.add_axes([xstart+i*figx, 0.75, figx, figy-yspace])
        plot_ICweight(ax, ne.patterns[i], thresh, direction='v', ylim=(ymin, ymax))
        if i > 0:
            ax.set_axis_off()
    
    # plot activities and raster for each neuron
    xstart = 0.06
    figx = (1-xstart)/n_ne
    xspace = 0.03
    centers = (ne.edges[:-1] + ne.edges[1:]) / 2
    activity_idx = np.where(ne.activity_alpha > activity_alpha-0.01)[0][0]
    for i in range(n_ne):
        
        if i not in ne.ne_members:
            continue
        # find the 0.5s with most ne spikes
        ne_spikes = ne.ne_units[i].spiketimes
        nspk, edges = np.histogram(ne_spikes, bins = ne.edges[::(2000//ne.df)])
        idx = np.argmax(nspk)
        t_start = edges[idx]
        t_end = edges[idx+1]
        
        # plot activity
        ax = fig.add_axes([xstart+i*figx, 0.65, figx-0.1, 0.05])
        ax.plot(centers, ne.ne_activity[i], color='k')
        activity_thresh = ne.activity_thresh[i][activity_idx]
        ax.plot([t_start, t_end], activity_thresh*np.array([1, 1]), color='r')
        ax.set_xlim([t_start, t_end])
        if i == 0:
            ymax = max(ne.ne_activity[i])
            ymin = min(ne.ne_activity[i])
            ax.set_ylabel('activity (a.u.)')

        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_ylim([ymin, ymax])
        
        # plot raster
        ax = fig.add_axes([xstart+i*figx, 0.35, figx-0.1, 0.25])
        # raster of all spikes from neurons
        plot_raster(ax, session.units)
        # raster of cNE events
        ax.eventplot(ne.ne_units[i].spiketimes, lineoffsets=n_neuron+1, linelengths=0.8, colors='r')
        # raster of ne spikes
        plot_raster(ax, ne.member_ne_spikes[i], offset='unit', color='r')
        # add scale bar
        ax.set_xlim([t_start, t_end])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        if i == 0:
            ax.plot([t_start, t_start+200], [0.1, 0.1], color='k', linewidth=5)
            ax.text(t_start+50, -0.6, '0.2 s')
            ax.set_yticks(list(range(5, n_neuron, 5)) + [n_neuron+1])
            ax.set_yticklabels(list(range(5, n_neuron, 5)) + ['cNE'])
            ax.set_ylabel('neuron #', labelpad=-10)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.set_ylim([0, n_neuron+1.5])
        ax.set_title('cNE #{}'.format(i+1))
        
    name_base = re.findall('\d{6}_\d{6}.*', session_file_path)
    name_base = re.sub(r'.pkl', r'.jpg', name_base[0])
    fig.savefig(os.path.join(savepath, name_base))
    plt.close(fig)
        
    

def plot_5ms_strf_ne_and_members(ne, savepath):
    xspace = 0.1
    yspace = 0.08
    figx = 0.25
    figy = 0.2
    
    # load session file
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


def plot_ICweight(ax, weights, thresh, direction='h', ylim=None):
    markersize=10
    n_neuron = len(weights)
    
    # plot threshold of membership
    if direction=='h':
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
        if any(members):
            markerline, _, _ = ax.stem(
                members+1, weights[members], 
                markerfmt='or', linefmt='r-', basefmt='k-')
            plt.setp(markerline, markersize = markersize)
        
        ax.set_xlim([0, n_neuron+1])
        ax.set_xticks(range(1, n_neuron+1, 5))
        ax.set_xlabel('neuron #')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel('ICweight')
    elif direction=='v':
        p = mpl.patches.Rectangle((-thresh, 0), 2*thresh, n_neuron+1, color='aquamarine')
        ax.add_patch(p)
        # stem plot for all weights of neurons
        markerline, stemline, baseline = ax.stem(
            range(1, n_neuron+1), weights, 
            markerfmt='ok', linefmt='k-', basefmt='k-', 
            orientation='horizontal')
        plt.setp(markerline, markersize = markersize)
        # plot baseline at 0
        ax.plot( [0, 0], [0, n_neuron+1], 'k-')
        
        # plot member stems
        members = np.where(weights > thresh)[0]
        if any(members):
            markerline, _, _ = ax.stem(
                members+1, weights[members], 
                markerfmt='or', linefmt='r-', basefmt='k-', 
                orientation='horizontal')
            plt.setp(markerline, markersize = markersize)
        
        ax.set_ylim([0, n_neuron+1])
        ax.set_yticks(range(1, n_neuron+1, 5))
        ax.set_ylabel('neuron #')
        if ylim:
            ax.set_xlim(ylim)
        ax.set_xlabel('ICweight')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

def plot_corrmat(ax, corr_mat):
    
    n_neuron = corr_mat.shape[0]
    np.fill_diagonal(corr_mat, 0)
    
    max_val = abs(corr_mat).max()*1.01
    ax.imshow(corr_mat, aspect='auto', origin='lower', cmap='RdBu_r', 
                  vmin=-max_val, vmax=max_val)
    
    ax.set_xticks(range(4, n_neuron, 5))
    ax.set_xticklabels(range(5, n_neuron+1, 5))
    ax.set_xlabel('neuron #')
    ax.set_yticks(range(4, n_neuron, 5))
    ax.set_yticklabels(range(5, n_neuron+1, 5))
    ax.set_ylabel('neuron #')


def plot_eigen_values(ax, corr_mat, thresh):
    
    n_neuron = corr_mat.shape[0]
    eigvals, _ = np.linalg.eig(corr_mat)
    eigvals.sort()
    eigvals = eigvals[::-1]
    ax.plot(range(1, n_neuron+1), eigvals, 'ko')
    ax.plot([0, n_neuron+1], [thresh, thresh], 'r--')
    ax.set_xticks(range(5, n_neuron+1, 5))
    ax.set_xlim([0, n_neuron+1])
    ax.set_xlabel('PC #')
    ax.set_ylabel('eigenvalue')
    

def plot_ICweigh_imshow(ax, patterns, members):
     patterns = np.transpose(patterns)
     max_val = abs(patterns).max()*1.01
     ax.imshow(patterns, aspect='auto', origin='lower', cmap='RdBu_r', 
                    vmin=-max_val, vmax=max_val)
     
     # highlight members
     for idx, member in members.items():
         ax.scatter([idx]*len(member), member, c='g')
     ax.set_xticks(range(patterns.shape[1]))
     ax.set_xticklabels(range(1, patterns.shape[1]+1))
     ax.set_xlabel('IC #')
     ax.set_yticks(range(4, patterns.shape[0], 5))
     ax.set_yticklabels(range(5, patterns.shape[0]+1, 5))
     ax.set_ylabel('neuron #')


def plot_raster(ax, units, offset='idx', color='k'):
    for idx, unit in enumerate(units):
        if offset == 'idx':
            pass
        elif offset == 'unit':
            idx = unit.unit
        ax.eventplot(unit.spiketimes, lineoffsets=idx+1, linelengths=0.8, colors=color)
   