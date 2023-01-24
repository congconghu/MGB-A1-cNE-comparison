# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:47:51 2022

@author: Congcong
"""
import re
import pickle
import os
import glob
import random
import math

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker, HPacker
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
from scipy.ndimage.filters import convolve
import ne_toolbox as netools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from helper import get_A1_MGB_files, get_distance, chi_square

plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1
mpl.rcParams['font.family'] = 'Arial'
cm = 1/2.54  # centimeters in inches
figure_size = [(17.6*cm, 17*cm), (11.6*cm, 17*cm), (8.5*cm, 17*cm)]
activity_alpha = 99.5
colors = sns.color_palette("Paired")
A1_color = (colors[1], colors[0])
MGB_color = (colors[5], colors[4])
colors_split = [colors[i] for i in [3, 2, 7, 6, 9, 8]]


# -------------------------------------- single unit properties -------------------------------------------------------
def plot_strf_df(units, figfolder, order='strf_ri', properties=False, smooth=False):
    if order == 'strf_ri':
        order_idx = (units[order]
                     .apply(np.mean)
                     .sort_values(ascending=False))
    elif order == 'strf_ri_p':
        order_idx = units[order].sort_values(ascending=True)
    elif order in ('strf_ri_z', 'strf_ptd'):
        order_idx = units[order].sort_values(ascending=False)
    elif  order == 'strf_info':
        order_idx = units[order].apply(np.mean).sort_values(ascending=False)

    # plot all units sorted by order
    fig, axes = plt.subplots(ncols=4, nrows=5, figsize=figure_size[0])
    axes = axes.flatten()
    c = 0
    nfig = 0
    for i, val in zip(order_idx.index, order_idx.values):
        strf = np.array(units.iloc[i].strf)
        axes[c].clear()
        if properties:
            plot_strf(axes[c],
                      strf,
                      taxis=np.array(units.iloc[i].strf_taxis),
                      faxis=np.array(units.iloc[i].strf_faxis),
                      latency=np.array(units.iloc[i].latency),
                      bf=np.array(units.iloc[i].bf),
                      smooth=smooth)
        else:
            plot_strf(axes[c],
                      strf,
                      taxis=np.array(units.iloc[i].strf_taxis),
                      faxis=np.array(units.iloc[i].strf_faxis),
                      smooth=smooth)
        axes[c].set_title('{:.3f}'.format(val))
        c += 1
        if c == 20:
            plt.tight_layout()
            fig.savefig(os.path.join(figfolder, '{}-{}.jpg'.format(order, nfig)), dpi=300)
            nfig += 1
            c = 0
    if c != 20:
        for i in range(c, 20):
            axes[i].remove()
            fig.savefig(os.path.join(figfolder, '{}-{}.jpg'.format(order, nfig)), dpi=300)


def plot_crh_df(units, figfolder, order='strf_ri', properties=False):
    if order == 'crh_ri':
        order_idx = (units[order]
                     .apply(np.mean)
                     .sort_values(ascending=False))
    elif order == 'crh_ri_p':
        order_idx = units[order].sort_values(ascending=True)
    elif order in ('crh_ri_z', 'crh_morani'):
        order_idx = units[order].sort_values(ascending=False)

    # plot all units sorted by order
    fig, axes = plt.subplots(ncols=4, nrows=5, figsize=figure_size[0])
    axes = axes.flatten()
    c = 0
    nfig = 0
    for i, val in zip(order_idx.index, order_idx.values):
        crh = np.array(units.iloc[i].crh)
        axes[c].clear()
        if properties:
            plot_crh(axes[c],
                     crh,
                     tmfaxis=np.array(units.iloc[i].tmfaxis),
                     smfaxis=np.array(units.iloc[i].smfaxis),
                     btmf=units.iloc[i].btmf,
                     bsmf=units.iloc[i].bsmf)
        else:
            plot_crh(axes[c],
                     crh,
                     tmfaxis=np.array(units.iloc[i].tmfaxis),
                     smfaxis=np.array(units.iloc[i].smfaxis))
        axes[c].set_title('{:.3f}'.format(val))
        c += 1
        if c == 20:
            plt.tight_layout()
            fig.savefig(os.path.join(figfolder, '{}-{}.jpg'.format(order, nfig)), dpi=300)
            nfig += 1
            c = 0
    if c != 20:
        for i in range(c, 20):
            axes[i].remove()
            fig.savefig(os.path.join(figfolder, '{}-{}.jpg'.format(order, nfig)), dpi=300)


def plot_strf_nonlinearity_df(units, figfolder):
    order_idx = units['nonlin_asi'].sort_values(ascending=False)

    # plot all units sorted by order
    fig, axes = plt.subplots(ncols=4, nrows=5, figsize=figure_size[0])
    axes = axes.flatten()
    c = 0
    nfig = 0
    for i, val in zip(order_idx.index, order_idx.values):
        strf = np.array(units.iloc[i].strf)
        centers = np.array(units.iloc[i].nonlin_centers)
        fr = np.array(units.iloc[i].nonlin_fr)
        fr_mean = np.array(units.iloc[i].nonlin_fr_mean)
        # plot strf
        axes[c].clear()
        plot_strf(axes[c],
                  strf,
                  taxis=np.array(units.iloc[i].strf_taxis),
                  faxis=np.array(units.iloc[i].strf_faxis))
       
        c += 1
        axes[c].clear()
        plot_nonlinearity(axes[c], centers, fr, fr_mean)
        axes[c].set_title('{:.2f}'.format(val))
        c += 1
        if c == 20:
            plt.tight_layout()
            fig.savefig(os.path.join(figfolder, 'nonlineairty-{}.jpg'.format(nfig)), dpi=300)
            nfig += 1
            c = 0
    if c != 20:
        for i in range(c, 20):
            axes[i].remove()
            fig.savefig(os.path.join(figfolder, 'nonlinearity-{}.jpg'.format(nfig)), dpi=300)

def plot_strf(ax, strf, taxis, faxis, latency=None, bf=None, smooth=False):
    """
    plot strf and format axis labels for strf

    Input
        ax: axes to plot on
        strf: matrix
        taxis: time axis for strf
        faxis: frequency axis for strf
    """
    strf = np.array(strf)
    taxis = np.array(taxis)
    faxis = np.array(faxis)
    max_val = abs(strf).max() * 1.01
    if smooth:
        weights = np.array([[1],
                            [2],
                            [1]],
                           dtype=np.float)
        weights = weights / np.sum(weights[:])
        strf = convolve(strf, weights, mode='constant')
    ax.imshow(strf, aspect='auto', origin='lower', cmap='RdBu_r',
              vmin=-max_val, vmax=max_val)

    tlabels = np.array([75, 50, 25, 0])
    xticks = np.searchsorted(-taxis, -tlabels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(tlabels)
    ax.set_xlabel('time before spike (ms)')

    faxis = faxis / 1000
    flabels = ['0.5', '2', '8', '32']
    flabels_arr = np.array([0.5, 2, 8, 32])
    yticks = np.searchsorted(faxis, flabels_arr)
    ax.set_yticks(yticks)
    ax.set_yticklabels(flabels)
    ax.set_ylabel('frequency (kHz)', labelpad=-2)

    if bf and latency is not None:
        idx_t = np.where(taxis <= latency)[0][0]
        idx_f = np.where(faxis >= bf / 1000)[0][0]
        ax.plot([0, idx_t], [idx_f, idx_f], 'k--')
        ax.plot([idx_t, idx_t], [0, idx_f], 'k--')


def plot_crh(ax, crh, tmfaxis, smfaxis, btmf=None, bsmf=None):
    """
    plot crh and format axis labels for crh

    Input
        ax: axes to plot on
        crh: matrix
        tmfaxis: temporal modulation frequency axis for crh
        smfaxis: spectral modulation frequency axis for crh
    """
    max_val = abs(crh).max() * 1.01
    ax.imshow(crh, aspect='auto', origin='lower', cmap='Spectral_r',
              vmin=0, vmax=max_val)

    tlabels = np.array([-40, -20, 0, 20, 40])
    ax.set_xticks(range(0, len(tmfaxis), 4))
    ax.set_xticklabels(tlabels)
    ax.set_xlabel('TMF (Hz)')

    flabels = np.array(range(5))
    ax.set_yticks(range(0, len(smfaxis), 4))
    ax.set_yticklabels(flabels)
    ax.set_ylabel('SMF (cyc/oct)', labelpad=0)
    if btmf is not None and bsmf is not None:
        idx_t = np.where(tmfaxis <= btmf)[0][-1]
        idx_f = np.where(smfaxis <= bsmf)[0][-1] + 0.5
        ax.text(idx_t, idx_f, '*', color='k', fontsize=30, ha='center', va='center')

def plot_nonlinearity(ax, centers, fr, fr_mean):
    
    ax.plot(centers, fr, 'ko-', ms=3)
    ax.plot([centers[0]-1, centers[-1]+1], [fr_mean, fr_mean], 'k--')
    ax.set_xlabel('similarity (s.d.)')
    ax.set_ylabel('fr (spk/s)')
    
    
# -------------------------------cNE method illustration ---------------------------------------
def plot_ne_construction(ne, savepath):
    """
    cNE construction procedure
    """
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
    ax = fig.add_axes([xspace, 0.75, figx - xspace, figy - yspace])
    corr_mat = np.corrcoef(ne.spktrain)
    plot_corrmat(ax, corr_mat)

    # plot eigenvalues
    ax = fig.add_axes([xspace + figx, 0.75, figx - xspace, figy - yspace])
    corr_mat = np.corrcoef(ne.spktrain)
    thresh = netools.get_pc_thresh(ne.spktrain)
    plot_eigen_values(ax, corr_mat, thresh)

    # plot IC
    ax = fig.add_axes([xspace + 2 * figx, 0.75, figx - xspace, figy - yspace])
    plot_ICweigh_imshow(ax, ne.patterns, ne.ne_members)

    # plot each IC
    n_ne, n_neuron = ne.patterns.shape
    xstart = 3 * figx + xspace
    figx = (1 - xstart) / n_ne
    thresh = 1 / np.sqrt(n_neuron)
    ymax = ne.patterns.max() * 1.1
    ymin = ne.patterns.min() * 1.1
    for i in range(n_ne):
        ax = fig.add_axes([xstart + i * figx, 0.75, figx, figy - yspace])
        plot_ICweight(ax, ne.patterns[i], thresh, direction='v', ylim=(ymin, ymax))
        if i > 0:
            ax.set_axis_off()

    # plot activities and raster for each neuron
    xstart = 0.06
    figx = (1 - xstart) / n_ne
    xspace = 0.03
    centers = (ne.edges[:-1] + ne.edges[1:]) / 2
    activity_idx = np.where(ne.activity_alpha > activity_alpha - 0.01)[0][0]
    for i in range(n_ne):

        if i not in ne.ne_members:
            continue
        # find the 0.5s with most ne spikes
        ne_spikes = ne.ne_units[i].spiketimes
        nspk, edges = np.histogram(ne_spikes, bins=ne.edges[::(2000 // ne.df)])
        idx = np.argmax(nspk)
        t_start = edges[idx]
        t_end = edges[idx + 1]

        # plot activity
        ax = fig.add_axes([xstart + i * figx, 0.65, figx - 0.1, 0.05])
        ax.plot(centers, ne.ne_activity[i], color='k')
        activity_thresh = ne.activity_thresh[i][activity_idx]
        ax.plot([t_start, t_end], activity_thresh * np.array([1, 1]), color='r')
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
        ax = fig.add_axes([xstart + i * figx, 0.35, figx - 0.1, 0.25])
        # raster of all spikes from neurons
        plot_raster(ax, session.units)
        # raster of cNE events
        ax.eventplot(ne.ne_units[i].spiketimes, lineoffsets=n_neuron + 1, linelengths=0.8, colors='r')
        # raster of ne spikes
        plot_raster(ax, ne.member_ne_spikes[i], offset='unit', color='r')
        # add scale bar
        ax.set_xlim([t_start, t_end])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        if i == 0:
            ax.plot([t_start, t_start + 200], [0.1, 0.1], color='k', linewidth=5)
            ax.text(t_start + 50, -0.6, '0.2 s')
            ax.set_yticks(list(range(5, n_neuron, 5)) + [n_neuron + 1])
            ax.set_yticklabels(list(range(5, n_neuron, 5)) + ['cNE'])
            ax.set_ylabel('neuron #', labelpad=-10)
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.set_ylim([0, n_neuron + 1.5])
        ax.set_title('cNE #{}'.format(i + 1))

    name_base = re.findall('\d{6}_\d{6}.*', session_file_path)
    name_base = re.sub(r'.pkl', r'.jpg', name_base[0])
    fig.savefig(os.path.join(savepath, name_base))
    plt.close(fig)


def plot_5ms_strf_ne_and_members(ne, savepath, ri=False, ri_z=False, freq=False):
    """
    cNE and member neuronn strf at 5ms resolution. use for sanity check
    """
    xspace = 0.1
    yspace = 0.08
    figx = 0.25
    figy = 0.2

    # load session file
    session_file_path = re.sub('-ne-.*.pkl', r'.pkl', ne.file_path)
    with open(session_file_path, 'rb') as f:
        session = pickle.load(f)
    n_neuron = ne.spktrain.shape[0]
    member_thresh = 1 / np.sqrt(n_neuron)  # threshol for membership

    if freq:
        ne_freq = pd.read_json('E:\Congcong\Documents\data\comparison\data-summary\cNE.json')
    for i, cne in enumerate(ne.ne_units):

        fig = plt.figure(figsize=figure_size)

        # plot cNE strf
        ax = fig.add_axes([xspace / 2, figy * 4 + yspace / 2, figx - xspace, figy - yspace])
        plot_strf(ax, cne.strf, taxis=cne.strf_taxis, faxis=cne.strf_faxis)
        if ri:
            ax.set_title('RI: {:.2f}'.format(np.mean(cne.strf_ri)))
        elif ri_z:
            ax.set_title('RI: {:.1f}'.format(np.mean(cne.strf_ri_z)))
        # plot ICweight
        idx_ne = cne.unit
        weights = ne.patterns[idx_ne]
        ax = fig.add_axes([xspace / 2 + figx, figy * 4 + yspace / 2, 2 * figx - xspace, figy - yspace])
        plot_ICweight(ax, weights, member_thresh)
        if freq:
            exp = session.exp.replace("_", "")
            freq_span = ne_freq[(ne_freq.exp == np.int(exp)) & (ne_freq.probe == session.probe)
                                & (ne_freq.cNE == i)]['freq_span_oct'].values[0]
            ax.set_title('{:.2f}'.format(freq_span))

        # plot member strf
        members = ne.ne_members[idx_ne]
        for idx, member in enumerate(members):
            unit = session.units[member]
            nrow = 3 - idx // 4
            ncol = idx % 4
            ax = fig.add_axes([xspace / 2 + figx * ncol, figy * nrow + yspace / 2,
                               figx - xspace, figy - yspace])
            plot_strf(ax, unit.strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis)
            if ri:
                ax.text(2, 50, '{:.2f}'.format(np.mean(unit.strf_ri)))
            elif ri_z:
                ax.set_title('{:.1f}'.format(np.mean(unit.strf_ri_z)))
            ax.set_title('unit {}'.format(unit.unit))
        name_base = re.findall('\d{6}_\d{6}.*', session_file_path)
        name_base = re.sub('.pkl', '-cNE_{}.jpg'.format(idx_ne), name_base[0])
        fig.savefig(os.path.join(savepath, name_base))
        plt.close(fig)


def plot_crh_ne_and_members(ne, savepath, ri=False):
    """
    cNE and member neuronn strf at 5ms resolution. use for sanity check
    """
    xspace = 0.1
    yspace = 0.08
    figx = 0.25
    figy = 0.2

    # load session file
    session_file_path = re.sub('-ne-.*.pkl', r'.pkl', ne.file_path)
    with open(session_file_path, 'rb') as f:
        session = pickle.load(f)
    n_neuron = ne.spktrain.shape[0]
    member_thresh = 1 / np.sqrt(n_neuron)  # threshol for membership

    for cne in ne.ne_units:

        fig = plt.figure(figsize=figure_size)

        # plot cNE strf
        ax = fig.add_axes([xspace / 2, figy * 4 + yspace / 2, figx - xspace, figy - yspace])
        plot_crh(ax, cne.crh, tmfaxis=cne.tmfaxis, smfaxis=cne.smfaxis)
        if ri:
            ax.set_title('RI: {:.2f}'.format(np.mean(cne.strf_ri)))
        # plot ICweight
        idx_ne = cne.unit
        weights = ne.patterns[idx_ne]
        ax = fig.add_axes([xspace / 2 + figx, figy * 4 + yspace / 2, 2 * figx - xspace, figy - yspace])
        plot_ICweight(ax, weights, member_thresh)

        # plot member strf
        members = ne.ne_members[idx_ne]
        for idx, member in enumerate(members):
            unit = session.units[member]
            nrow = 3 - idx // 4
            ncol = idx % 4
            ax = fig.add_axes([xspace / 2 + figx * ncol, figy * nrow + yspace / 2,
                               figx - xspace, figy - yspace])
            plot_crh(ax, unit.crh, tmfaxis=unit.tmfaxis, smfaxis=unit.smfaxis)

            if ri:
                ax.text(2, 50, '{:.2f}'.format(np.mean(unit.strf_ri)))
            ax.set_title('unit {}'.format(unit.unit))
        name_base = re.findall('\d{6}_\d{6}.*', session_file_path)
        name_base = re.sub('.pkl', '-cNE_{}.jpg'.format(idx_ne), name_base[0])
        fig.savefig(os.path.join(savepath, name_base))
        plt.close(fig)


def plot_ICweight(ax, weights, thresh, direction='h', ylim=None):
    """
    stem plot for cNE patterns
    """
    markersize = 10
    n_neuron = len(weights)

    # plot threshold of membership
    if direction == 'h':
        ax.plot([0, n_neuron + 1], [thresh, thresh], 'r--')
        # stem plot for all weights of neurons
        markerline, stemline, baseline = ax.stem(
            range(1, n_neuron + 1), weights,
            markerfmt='ok', linefmt='k-', basefmt='k-')
        plt.setp(markerline, markersize=markersize)
        # plot baseline at 0
        ax.plot([0, n_neuron + 1], [0, 0], 'k-')

        # plot member stems
        members = np.where(weights > thresh)[0]
        if any(members):
            markerline, _, _ = ax.stem(
                members + 1, weights[members],
                markerfmt='or', linefmt='r-', basefmt='k-')
            plt.setp(markerline, markersize=markersize)

        ax.set_xlim([0, n_neuron + 1])
        ax.set_xticks(range(5, n_neuron + 1, 5))
        ax.set_xlabel('neuron #')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel('ICweight')
    elif direction == 'v':
        p = mpl.patches.Rectangle((-thresh, 0), 2 * thresh, n_neuron + 1, color='aquamarine')
        ax.add_patch(p)
        # stem plot for all weights of neurons
        markerline, stemline, baseline = ax.stem(
            range(1, n_neuron + 1), weights,
            markerfmt='ok', linefmt='k-', basefmt='k-',
            orientation='horizontal')
        plt.setp(markerline, markersize=markersize)
        # plot baseline at 0
        ax.plot([0, 0], [0, n_neuron + 1], 'k-')

        # plot member stems
        members = np.where(weights > thresh)[0]
        if any(members):
            markerline, _, _ = ax.stem(
                members + 1, weights[members],
                markerfmt='or', linefmt='r-', basefmt='k-',
                orientation='horizontal')
            plt.setp(markerline, markersize=markersize)

        ax.set_ylim([0, n_neuron + 1])
        ax.set_yticks(range(5, n_neuron + 1, 5))
        ax.set_ylabel('neuron #')
        if ylim:
            ax.set_xlim(ylim)
        ax.set_xlabel('ICweight')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def plot_corrmat(ax, corr_mat):
    """
    heatmap of correlation matrix of neurons
    """
    n_neuron = corr_mat.shape[0]
    np.fill_diagonal(corr_mat, 0)

    max_val = abs(corr_mat).max() * 1.01
    im = ax.imshow(corr_mat, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-max_val, vmax=max_val)

    ax.set_xticks(range(4, n_neuron, 5))
    ax.set_xticklabels(range(5, n_neuron + 1, 5))
    ax.set_xlabel('neuron #')
    ax.set_yticks(range(4, n_neuron, 5))
    ax.set_yticklabels(range(5, n_neuron + 1, 5))
    ax.set_ylabel('neuron #')
    return im


def plot_eigen_values(ax, corr_mat, thresh):
    """
    scatter plot of eigen values, with Marcenko-Pastur threshold
    """
    n_neuron = corr_mat.shape[0]
    eigvals, _ = np.linalg.eig(corr_mat)
    eigvals.sort()
    eigvals = eigvals[::-1]
    ax.plot(range(1, n_neuron + 1), eigvals, 'ko')
    ax.plot([0, n_neuron + 1], [thresh, thresh], 'r--')
    ax.set_xticks(range(5, n_neuron + 1, 5))
    ax.set_xlim([0, n_neuron + 1])
    ax.set_xlabel('PC #')
    ax.set_ylabel('eigenvalue')


def plot_ICweigh_imshow(ax, patterns, members):
    """heatmap of patterns, with member neurons highlighted"""
    patterns = np.transpose(patterns)
    max_val = abs(patterns).max() * 1.01
    ax.imshow(patterns, aspect='auto', origin='lower', cmap='RdBu_r',
              vmin=-max_val, vmax=max_val)

    # highlight members
    for idx, member in members.items():
        ax.scatter([idx] * len(member), member, c='aquamarine')
    ax.set_xticks(range(patterns.shape[1]))
    ax.set_xticklabels(range(1, patterns.shape[1] + 1))
    ax.set_xlabel('IC #')
    ax.set_yticks(range(4, patterns.shape[0], 5))
    ax.set_yticklabels(range(5, patterns.shape[0] + 1, 5))
    ax.set_ylabel('neuron #')


def plot_activity(ax, centers, activity, thresh, t_window, ylim):
    """plot activity with threshold"""
    ax.plot(centers, activity, color='k')
    ax.plot(t_window, thresh * np.array([1, 1]), color='r')
    ax.set_xlim(t_window)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim(ylim)


def plot_raster(ax, units, offset='idx', color='k', new_order=None):
    """raster plot of activities of member neurons"""
    for idx, unit in enumerate(units):
        if offset == 'idx':
            pass
        elif offset == 'unit':
            idx = unit.unit
        if new_order is not None:
            idx = new_order[idx]
        ax.eventplot(unit.spiketimes, lineoffsets=idx + 1, linelengths=0.8, colors=color)


def figure1(datafolder, figfolder):
    """
    Figure1: groups of neurons with coordinated activities exist in A1 and MGB
    Plot construction procedures for cNEs and  correlated firing around cNE events
    cNE members show significantly higher cross correlation

    Input:
        datafolder: path to *ne-20dft-dmr.pkl files
        figfolder: path to save figure
    Return:
        None
    """

    # use example recording to plot construction procedure
    ne_file = os.path.join(datafolder, '200730_015848-site4-6200um-20db-dmr-30min-H31x64-fs20000-ne-20dft-dmr.pkl')
    with open(ne_file, 'rb') as f:
        ne = pickle.load(f)
    session_file = re.sub('-ne-20dft-dmr', '', ne.file_path)
    with open(session_file, 'rb') as f:
        session = pickle.load(f)

    fig = plt.figure(figsize=figure_size)

    # positions for first 3 plots
    y = 0.85
    figy = 0.10
    xstart = 0.05
    xspace = 0.05
    figx = 0.12

    # plot correlation matrix
    ax = fig.add_axes([xstart, y, figx, figy])
    corr_mat = np.corrcoef(ne.spktrain)
    im = plot_corrmat(ax, corr_mat)
    axins = inset_axes(
        ax,
        width="10%",  # width: 5% of parent_bbox width
        height="80%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=axins)
    cb.ax.tick_params(axis='y', direction='in')
    axins.set_title('  corr', fontsize=16)
    cb.ax.set_yticks([-0.1, 0, 0.1])
    cb.ax.set_yticklabels(['-0.1', '0', '0.1'])
    axins.tick_params(axis='both', which='major', labelsize=16)

    # plot eigen values
    ax = fig.add_axes([xstart + figx + xspace + 0.04, y, figx, figy])
    corr_mat = np.corrcoef(ne.spktrain)
    thresh = netools.get_pc_thresh(ne.spktrain)
    plot_eigen_values(ax, corr_mat, thresh)

    # plot ICweights - color coded
    ax = fig.add_axes([xstart + figx * 2 + xspace * 2 + 0.04, y, figx, figy])
    plot_ICweigh_imshow(ax, ne.patterns, ne.ne_members)

    # stem plots for ICweights
    xstart = 0.6
    xspace = 0.01
    figx = 0.08
    n_ne, n_neuron = ne.patterns.shape
    thresh = 1 / np.sqrt(n_neuron)
    member_labels = 'edcbaihgf'
    c = 0
    for i in range(4):
        ax = fig.add_axes([xstart + figx * i + xspace * i, y, figx, figy])
        plot_ICweight(ax, ne.patterns[i], thresh, direction='v', ylim=(-0.3, 0.8))
        if i > 0:
            ax.set_axis_off()
        if i < 2:
            members = ne.ne_members[i]
            for member in members:
                if i == 0:
                    ax.text(ne.patterns[i][member] + 0.08, member, member_labels[c],
                            fontsize=16)
                else:
                    ax.text(ne.patterns[i][member] + 0.08, member + 0.7, member_labels[c],
                            fontsize=16)
                c += 1

    # second row: activities
    y = 0.73
    figy = 0.05
    xstart = 0.05
    xspace = 0.02
    figx = 0.14
    centers = (ne.edges[:-1] + ne.edges[1:]) / 2
    activity_idx = 5  # 99.5% as threshold
    # reorder units
    new_order = np.array([0, 10, 11, 12, 1, 6, 7, 8, 9, 13, 2, 14, 3, 4, 5])
    c = 0
    for i in range(2):

        # find the 0.5s with most ne spikes
        ne_spikes = ne.ne_units[i].spiketimes
        nspk, edges = np.histogram(ne_spikes, bins=ne.edges[::(2000 // ne.df)])
        idx = np.argmax(nspk)
        t_start = edges[idx]
        t_end = edges[idx + 1]

        # plot activity
        ax = fig.add_axes([xstart + xspace * i + figx * i, y, figx, figy])
        activity_thresh = ne.activity_thresh[i][activity_idx]
        ylim = [-10, 50]
        plot_activity(ax, centers, ne.ne_activity[i], activity_thresh, [t_start, t_end], ylim)
        if i == 0:
            ax.set_ylabel('activity (a.u.)')
        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.set_title('cNE #{}'.format(i + 1), fontsize=18, fontweight='bold')

        # plot raster
        ax = fig.add_axes([xstart + xspace * i + figx * i, y - 0.25, figx, 0.245])
        members = ne.ne_members[i]
        for member in members:
            member_idx = new_order[member]
            p = mpl.patches.Rectangle((t_start, member_idx + 0.6),
                                      t_end - t_start, 0.8, color='aquamarine')
            ax.add_patch(p)
            ax.text(t_end, member_idx + 1, member_labels[c], color='r')
            c += 1

        plot_raster(ax, session.units, new_order=new_order)
        ax.eventplot(ne.ne_units[i].spiketimes, lineoffsets=n_neuron + 1, linelengths=0.8, colors='r')
        plot_raster(ax, ne.member_ne_spikes[i], offset='unit', color='r', new_order=new_order)
        ax.set_xlim([t_start, t_end])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])

        if i == 0:
            # scale bar
            ax.plot([t_start, t_start + 200], [0.1, 0.1], color='k', linewidth=5)
            ax.text(t_start, -0.8, '0.2 s')
            ax.set_yticks([n_neuron + 1])
            ax.tick_params(axis='y', length=0)
            ax.set_yticklabels(['cNE'], fontsize=16, color='r')
        else:
            ax.set_yticks([])
        ax.set_ylim([0, n_neuron + 1.5])

    # xcorr plot
    xstart = 0.42
    figx = 0.15
    xspace = 0.02
    y = 0.5
    figy = 0.06
    yspace = 0.01
    ax = []
    for i in range(4):
        y_extra = 0.02 if i < 2 else 0
        for j in range(2):
            ax.append(fig.add_axes([xstart + j * figx + j * xspace,
                                    y + (3 - i) * figy + (3 - i) * yspace + y_extra,
                                    figx, figy]))
    ax = np.reshape(np.array(ax), (4, 2))
    xcorr = pd.read_json(
        r'E:\Congcong\Documents\data\comparison\data-summary\member_nonmember_pair_xcorr_filtered.json')
    xcorr = xcorr[xcorr['stim'] == 'dmr']
    plot_xcorr(fig, ax, xcorr.groupby(by=['region', 'member']))

    # boxplots
    xstart = 0.83
    y = 0.69
    figx = 0.12
    figy = 0.10
    my_order = ['MGB_dmr_(w)', 'MGB_dmr_(o)', 'A1_dmr_(w)', 'A1_dmr_(o)']

    # box plot for peak value
    ax = fig.add_axes([xstart, y, figx, figy])
    peak_mean = xcorr[xcorr['xcorr_sig']].groupby(
        by=['exp', 'region', 'stim', 'member'], as_index=False)['peak'].mean()
    peak_mean['region_stim_member'] = peak_mean[['region', 'stim', 'member']].apply(tuple, axis=1)
    peak_mean['region_stim_member'] = peak_mean['region_stim_member'].apply(lambda x: '_'.join([str(y) for y in x]))
    boxplot_scatter(ax=ax, x='region_stim_member', y='peak', data=peak_mean, order=my_order,
                    hue='region_stim_member', palette=list(MGB_color) + list(A1_color), hue_order=my_order,
                    jitter=0.4, legend=False)
    ax.set_yticks(range(0, 7, 2))
    ax.set_ylabel('mean z-scored\nCCG peak value', labelpad=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['MGB\n(w)', 'MGB\n(o)', 'A1\n(w)', 'A1\n(o)'], fontsize=16)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([0, 7])
    ax.set_xlabel('')
    # significance test for z-scored CCG peak value: within vs outside cNE
    for i in range(2):
        res = stats.mannwhitneyu(x=peak_mean[peak_mean['region_stim_member'] == my_order[i * 2]]['peak'],
                                 y=peak_mean[peak_mean['region_stim_member'] == my_order[i * 2 + 1]]['peak'],
                                 alternative='greater')
        p = res.pvalue
        plot_significance_star(ax, p, [i * 2, i * 2 + 1], 6.2, 6.3)
        # significance test for z-scored CCG peak value: A1 vs MGB
    for i in range(2):
        res = stats.mannwhitneyu(x=peak_mean[peak_mean['region_stim_member'] == my_order[i]]['peak'],
                                 y=peak_mean[peak_mean['region_stim_member'] == my_order[i + 2]]['peak'])
        p = res.pvalue
        plot_significance_star(ax, p, [i, i + 2], 6.5, 6.6)

    # box plot for correlation value
    ax = fig.add_axes([xstart, y - 0.16, figx, figy])
    boxplot_scatter(ax=ax, x='region_stim_member', y='corr', data=xcorr, order=my_order,
                    hue='region_stim_member', palette=list(MGB_color) + list(A1_color), hue_order=my_order,
                    size=1, alpha=0.5, jitter=0.4, legend=False)
    # significance test for correlation value: within vs outside cNE
    for i in range(2):
        res = stats.mannwhitneyu(x=xcorr[xcorr['region_stim_member'] == my_order[i * 2]]['corr'],
                                 y=xcorr[xcorr['region_stim_member'] == my_order[i * 2 + 1]]['corr'],
                                 alternative='greater')
        p = res.pvalue
        plot_significance_star(ax, p, [i * 2, i * 2 + 1], 0.2, 0.21)
    # significance test for correlation value: A1 vs MGB
    for i in range(2):
        res = stats.mannwhitneyu(x=xcorr[xcorr['region_stim_member'] == my_order[i]]['corr'],
                                 y=xcorr[xcorr['region_stim_member'] == my_order[i + 2]]['corr'])
        p = res.pvalue
        plot_significance_star(ax, p, [i, i + 2], 0.21 + 0.03 * i, 0.22 + 0.03 * i)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylabel('correlation', labelpad=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['MGB\n(w)', 'MGB\n(o)', 'A1\n(w)', 'A1\n(o)'], fontsize=16)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([-0.05, 0.3])
    ax.set_xlabel('')

    fig.savefig(os.path.join(figfolder, 'fig1.png'))
    fig.savefig(os.path.join(figfolder, 'fig1.svg'))
    fig.savefig(os.path.join(figfolder, 'fig1.pdf'))


def plot_xcorr(fig, ax, xcorr, savepath=None):
    """
    Summary plot of cross correlation of member and nonmember pairs under 2 stimulus conditions ans in 2 regions

    Input:
        fig: handle for the figure to be plotted on
        ax: list of axes
        xcorr: groupby object containing (region,member) conditions and cross correlation data
        savepath: folder path to save the summary plot
    Return:
        None
    """
    for key, data in xcorr:
        region, member = key
        row = 0 if region == 'MGB' else 2
        col = 0 if member == '(w)' else 1
        corr, im = plot_xcorr_imshow(ax[row][col], data)
        plot_xcorr_avg(ax[row + 1][col], corr)
    for row in range(4):
        for col in range(2):
            if col > 0:
                ax[row][col].set_ylabel('')

            if row == 3:
                ax[row][col].set_xticks(range(-200, 201, 100))
                ax[row][col].set_xticklabels(["", '', 0, '', ""])
                ax[row][col].set_xticks([-180, 180], minor=True)
                ax[row][col].set_xticklabels([-200, 200], minor=True)
                for line in ax[row][col].xaxis.get_minorticklines():
                    line.set_visible(False)
                ax[row][col].set_xlabel('lag (ms)')
            else:
                ax[row][col].set_xlabel('')

                if row in (0, 2):
                    ax[row][col].set_xlabel('')
                    ax[row][col].set_xticks(range(0, 401, 100))
                    t = ax[row][col].yaxis.get_offset_text()
                    t.set_x(-0.05)
                else:
                    ax[row][col].set_ylim([-1, 3])
                    ax[row][col].set_xticks(range(-200, 201, 100))
                ax[row][col].set_xticklabels([''] * 5)

    # titles
    ax[0][0].set_title('MGB (within cNE)', fontsize=16, color=MGB_color[0], fontweight="bold")
    ax[0][1].set_title('MGB (outside cNE)', fontsize=16, color=MGB_color[1], fontweight="bold")
    ax[2][0].set_title('A1 (within cNE)', fontsize=16, color=A1_color[0], fontweight="bold")
    ax[2][1].set_title('A1 (outside cNE)', fontsize=16, color=A1_color[1], fontweight="bold")
    # colorbar
    axins = inset_axes(
        ax[2][1],
        width="5%",  # width: 5% of parent_bbox width
        height="80%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax[2][1].transAxes,
        borderpad=0
    )
    cb = fig.colorbar(im, cax=axins)
    cb.ax.tick_params(axis='y', direction='in')
    axins.set_title('z-scored\ncorr\n', fontsize=16, linespacing=0.8)
    cb.ax.set_yticks([-5, 0, 5])
    cb.ax.set_yticklabels(['-5', '0', '5'])
    axins.tick_params(axis='both', which='major', labelsize=16)
    if savepath:
        fig.savefig(os.path.join(savepath, 'xcorr_member_nonmember.jpg'))
        plt.close(fig)


def plot_xcorr_imshow(ax, data):
    """Stack cross correlation curves and plot as heatmap"""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    corr = data['xcorr'].to_numpy()
    corr = np.stack(corr)
    corr = zscore(corr, axis=1)
    idx_peak = corr.argmax(axis=1)
    order = np.argsort(idx_peak)
    corr = corr[order]
    im = ax.imshow(corr, aspect='auto', origin='lower', vmax=5, vmin=-5, cmap='jet')
    ax.set_xticks(range(0, len(corr[0]), 100))
    ax.set_xticklabels(range(-200, 201, 100))
    ax.set_ylabel('neuron pair #')
    ax.set_xlabel('lag (ms)')
    return corr, im


def plot_xcorr_avg(ax, corr):
    """plot averaged cross correlations and standard deviation"""
    x = range(-200, 201)
    corr_avg = corr.mean(axis=0)
    corr_std = corr.std(axis=0)
    ax.fill_between(x, corr_avg - corr_std, corr_avg + corr_std,
                    alpha=0.5, edgecolor=None, facecolor='k')
    ax.plot(x, corr_avg, color='k')
    ax.plot([-200, 200], [0, 0], linewidth=1, color='k')
    ax.plot([0, 0], [-1, 3], linewidth=1, color='k')
    ax.set_ylim([-1, 3])
    ax.set_xlim([-200, 200])
    ax.set_ylabel('z-scored corr')


# --------------------------------------------------cNE properties ---------------------------------------------------
def num_ne_vs_num_neuron(ax, datafolder, stim):
    # get files for recordings in A1 and MGB
    files = get_A1_MGB_files(datafolder, stim)

    # get the number of neurons and the number of cNEs in each recording
    n_neuron = {'A1': [], 'MGB': []}
    n_ne = {'A1': [], 'MGB': []}
    for region, filepaths in files.items():
        for file in filepaths:
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            n_neuron[region].append(ne.patterns.shape[1])
            n_ne[region].append(len(ne.ne_members))
    sc = []
    for region in files.keys():
        sc.append(ax.scatter(n_neuron[region], n_ne[region], color=eval(region + '_color[0]')))
        a, b = np.polyfit(n_neuron[region], n_ne[region], deg=1)
        x = np.linspace(5, 45, num=10)
        ax.plot(x, a * x + b, color=eval(region + '_color[0]'))
    ax.legend(sc, ['A1', 'MGB'])
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('# of cNEs')


def ne_size_vs_num_neuron(ax, datafolder, stim, plot_type='mean', relative=False, n_neuron_filter=()):
    """
    stim: 'dmr' or 'spon'
    plot_type: 'raw', 'mean', 'median'
    relative: 'True' if plot boxplot of ratio, 'False' if plot scatter of raw size
    n_neuron_filter: filter to include recordings with neuron number within the range
        e.g. (10, 30) to only include recordings with >= 10 neurons and <= 30 neurons
    """
    # get files for recordings in A1 and MGB
    files = get_A1_MGB_files(datafolder, stim)

    # get the number of neurons and the number of cNEs in each recording
    n_neuron = {'A1': [], 'MGB': []}
    ne_size = {'A1': [], 'MGB': []}
    for region, filepaths in files.items():
        for file in filepaths:
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            if any(n_neuron_filter):
                if ne.patterns.shape[1] > n_neuron_filter[1] or ne.patterns.shape[1] < n_neuron_filter[0]:
                    continue
            # get size of each cNE
            n_members = [len(x) for x in ne.ne_members.values()]
            if plot_type == 'raw':
                n_neuron[region].extend([ne.patterns.shape[1]] * len(n_members))
                ne_size[region].extend(n_members)
            else:
                n_neuron[region].append(ne.patterns.shape[1])
                ne_size[region].append(eval('np.{}(n_members)'.format(plot_type)))
    if relative:
        ne_size_df = pd.DataFrame({'region': [], 'size': []})
        for region, size in ne_size.items():
            ne_size[region] = np.array(ne_size[region]) / np.array(n_neuron[region])
            ne_size_df = pd.concat([ne_size_df,
                                    pd.DataFrame({'region': [region] * len(ne_size[region]),
                                                  'size': ne_size[region]})],
                                   axis=0)
        boxplot_scatter(ax, x='region', y='size', data=ne_size_df,
                        order=('MGB', 'A1'), hue='region',
                        palette=(MGB_color[0], A1_color[0]), hue_order=('MGB', 'A1'))
        ax.set_xlabel('')
        ax.set_ylabel('cNE size')

        res = stats.mannwhitneyu(ne_size['MGB'], ne_size['A1'])
        p = res.pvalue
        plot_significance_star(ax, p, [0, 1], 0.55, 0.56)
        ax.set_ylim([0, 0.6])
    else:
        sc = []
        for region in files.keys():
            sc.append(ax.scatter(n_neuron[region], ne_size[region], color=eval(region + '_color[0]')))
            a, b = np.polyfit(n_neuron[region], ne_size[region], deg=1)
            x = np.linspace(5, 45, num=10)
            ax.plot(x, a * x + b, color=eval(region + '_color[0]'))
        ax.legend(sc, ['A1', 'MGB'])
        ax.set_xlabel('# of neurons')
        ax.set_ylabel('{} cNE size'.format(plot_type))


def num_ne_participate(ax, datafolder, stim, n_neuron_filter=()):
    files = get_A1_MGB_files(datafolder, stim)
    n_participation = {'A1': [], 'MGB': []}
    for region, filepaths in files.items():
        for file in filepaths:
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            if any(n_neuron_filter):
                if ne.patterns.shape[1] > n_neuron_filter[1] or ne.patterns.shape[1] < n_neuron_filter[0]:
                    continue
            n_neuron = ne.patterns.shape[1]
            participate = np.concatenate(list(ne.ne_members.values()))
            unique, counts = np.unique(participate, return_counts=True)
            participate = dict(zip(unique, counts))
            for i in range(n_neuron):
                if i in participate:
                    n_participation[region].append(participate[i])
                else:
                    n_participation[region].append(0)
    for region, participate in n_participation.items():
        ax.hist(n_participation[region], bins=range(6), alpha=0.5,
                label=region, density=True, align='left', color=eval(region + '_color[0]'))
    ax.legend()
    ax.set_xlabel('number of cNEs each neuron belonged to')
    ax.set_ylabel('ratio')


def ne_member_distance(ax, datafolder, stim, probe, direction='vertical'):
    files = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = [x for x in files if probe in x]
    dist_member = []
    dist_nonmember = []
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        member_pairs, nonmember_pairs = ne.get_member_pairs()
        session_file = re.sub('-ne-20dft-' + stim, '', file)
        with open(session_file, 'rb') as f:
            session = pickle.load(f)
        for pair in member_pairs:
            p1 = session.units[pair[0]].position
            p2 = session.units[pair[1]].position
            dist_member.append(get_distance(p1, p2, direction))
        for pair in nonmember_pairs:
            p1 = session.units[pair[0]].position
            p2 = session.units[pair[1]].position
            dist_nonmember.append(get_distance(p1, p2, direction))
    if probe == 'H31x64':
        color = MGB_color[0]
        step = 20
    else:
        color = A1_color[0]
        step = 25

    if direction == 'vertical':
        ax.hist(dist_member, range(0, 601, step), color=color,
                weights=np.repeat([1 / len(dist_member)], len(dist_member)))
        ax.hist(dist_nonmember, range(0, 601, step), color='k',
                weights=np.repeat([1 / len(dist_nonmember)], len(dist_nonmember)),
                fill=False, histtype='step')
        ax.set_xlabel('pairwise distance (um)')
        ax.set_xlim([0, 600])
    elif direction == 'horizontal':
        ax.hist(dist_member, 2, color=color, align='left', rwidth=0.8,
                weights=np.repeat([1 / len(dist_member)], len(dist_member)))
        ax.hist(dist_nonmember, 2, color='k', align='left', rwidth=0.8,
                weights=np.repeat([1 / len(dist_nonmember)], len(dist_nonmember)),
                fill=False)
        ax.set_xticks([0, 125])
        ax.set_xticklabels(['same shank', 'across shank'])
    ax.set_ylabel('ratio')


def ne_member_span(ax, datafolder, stim, probe):
    random.seed(0)
    files = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = [x for x in files if probe in x]
    span_member = []
    span_random = []
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        session_file = re.sub('-ne-20dft-' + stim, '', file)
        n_neuron = ne.patterns.shape[1]
        with open(session_file, 'rb') as f:
            session = pickle.load(f)

        for members in ne.ne_members.values():
            # get cNE span
            span_member.append(session.get_cluster_span(members))
            # get span of random neurons
            n_member = len(members)
            for _ in range(10):
                members_sham = random.sample(range(n_neuron), n_member)
                span_random.append(session.get_cluster_span(members_sham))

    if probe == 'H31x64':
        color = MGB_color[0]
        step = 20
    else:
        color = A1_color[0]
        step = 25
    ax.hist(span_member, range(0, 1001, step * 2), color=color,
            weights=np.repeat([1 / len(span_member)], len(span_member)))
    ax.hist(span_random, range(0, 1001, step * 2), color='k',
            weights=np.repeat([1 / len(span_random)], len(span_random)),
            fill=False, histtype='step')
    ax.set_ylabel('ratio')
    ax.set_xlabel('cNE span (um)')
    ax.set_xlim([0, 1000])


def ne_member_shank_span(ax, datafolder, stim, probe):
    random.seed(0)
    files = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = [x for x in files if probe in x]
    span_member = []
    span_random = []
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        session_file = re.sub('-ne-20dft-' + stim, '', file)
        n_neuron = ne.patterns.shape[1]
        with open(session_file, 'rb') as f:
            session = pickle.load(f)

        for members in ne.ne_members.values():
            # get cNE span
            span_member.append(session.get_cluster_span(members, 'horz'))
            # get span of random neurons
            n_member = len(members)
            for _ in range(10):
                members_sham = random.sample(range(n_neuron), n_member)
                span_random.append(session.get_cluster_span(members_sham, 'horz'))

    if probe == 'H31x64':
        color = MGB_color[0]
        step = 20
    else:
        color = A1_color[0]
        step = 25
    ax.hist(span_member, 2, color=color, align='left', rwidth=0.8,
            weights=np.repeat([1 / len(span_member)], len(span_member)))
    ax.hist(span_random, 2, color='k', align='left', rwidth=0.8,
            weights=np.repeat([1 / len(span_random)], len(span_random)),
            fill=False)
    ax.set_ylabel('ratio')
    ax.set_xticks([0, 125])
    ax.set_xticklabels(['same shank', 'across shank'])


def ne_freq_span_vs_all_freq_span(ax, datafolder, stim):
    files = get_A1_MGB_files(datafolder, stim)
    span_ne = {'A1': [], 'MGB': []}
    span_all = {'A1': [], 'MGB': []}
    for region, filepaths in files.items():
        for file in filepaths:
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            session_file = re.sub('-ne-20dft-' + stim, '', file)
            with open(session_file, 'rb') as f:
                session = pickle.load(f)
            n_neuron = ne.patterns.shape[1]
            for members in ne.ne_members.values():
                # get cNE frequency span
                span_ne[region].append(session.get_cluster_freq_span(members))
                # get frequency span of all neurons
                span_all[region].append(session.get_cluster_freq_span(range(n_neuron)))
    sc = []
    for region in files.keys():
        span_ne[region] = np.array(span_ne[region])
        span_all[region] = np.array(span_all[region])
        idx = ~ np.isnan(span_ne[region])
        span_ne[region] = span_ne[region][idx]
        span_all[region] = span_all[region][idx]
        sc.append(ax.scatter(span_all[region], span_ne[region], color=eval(region + '_color[0]')))
        a, b = np.polyfit(span_all[region], span_ne[region], deg=1)
        x = np.linspace(0, 6, num=10)
        ax.plot(x, a * x + b, color=eval(region + '_color[0]'))
        ax.legend(sc, ['A1', 'MGB'])
        ax.set_xlabel('all neurons')
        ax.set_ylabel('member neurons')


def ne_member_freq_span(ax, datafolder, stim, probe):
    random.seed(0)
    files = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = [x for x in files if probe in x]
    span_member = []
    span_random = []
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        session_file = re.sub('-ne-20dft-' + stim, '', file)
        n_neuron = ne.patterns.shape[1]
        with open(session_file, 'rb') as f:
            session = pickle.load(f)

        for members in ne.ne_members.values():
            # get cNE span
            span_member.append(session.get_cluster_freq_span(members))
            # get span of random neurons
            n_member = len(members)
            for _ in range(10):
                members_sham = random.sample(range(n_neuron), n_member)
                span_random.append(session.get_cluster_freq_span(members_sham))

    if probe == 'H31x64':
        color = MGB_color[0]
    else:
        color = A1_color[0]

    span_member = np.array(span_member)
    span_random = np.array(span_random)
    span_member = span_member[~np.isnan(span_member)]
    span_random = span_random[~np.isnan(span_random)]
    ax.hist(span_member, np.linspace(0, 6, 13), color=color,
            weights=np.repeat([1 / len(span_member)], len(span_member)))
    ax.hist(span_random, np.linspace(0, 6, 13), color='k',
            weights=np.repeat([1 / len(span_random)], len(span_random)),
            fill=False, histtype='step')
    ax.set_ylabel('ratio')
    ax.set_xlabel('frequency span (oct)')
    ax.set_xlim([0, 6])


def ne_member_freq_distance(ax, datafolder, stim, probe):
    files = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = [x for x in files if probe in x]
    dist_member = []
    dist_nonmember = []
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        member_pairs, nonmember_pairs = ne.get_member_pairs()
        session_file = re.sub('-ne-20dft-' + stim, '', file)
        with open(session_file, 'rb') as f:
            session = pickle.load(f)
        for pair in member_pairs:
            if session.units[pair[0]].strf_sig and session.units[pair[1]].strf_sig:
                f1 = session.units[pair[0]].bf
                f2 = session.units[pair[1]].bf
                dist_member.append(abs(math.log2(f1 / f2)))
        for pair in nonmember_pairs:
            if session.units[pair[0]].strf_sig and session.units[pair[1]].strf_sig:
                f1 = session.units[pair[0]].bf
                f2 = session.units[pair[1]].bf
                dist_nonmember.append(abs(math.log2(f1 / f2)))
    if probe == 'H31x64':
        color = MGB_color[0]
    else:
        color = A1_color[0]

    ax.hist(dist_member, np.linspace(0, 6, 25), color=color,
            weights=np.repeat([1 / len(dist_member)], len(dist_member)))
    ax.hist(dist_nonmember, np.linspace(0, 6, 25), color='k',
            weights=np.repeat([1 / len(dist_nonmember)], len(dist_nonmember)),
            fill=False, histtype='step')
    ax.set_xlabel('pairwise frequency difference (oct)')
    ax.set_xlim([0, 6])
    ax.set_ylabel('ratio')


# ---------------------plots for split ne dmr/spon stability-----------------------------
def plot_ne_split_ic_weight_match(ne_split, axes=None, figpath=None):
    """
    plot matching patterns for each cNE under 3 conditions: cross condition, spon and dmr

    Inputs:
        ne_split: dictionary containing ne data on 4 blocks
        figpath: file path to save figures
    """

    n_dmr = len(ne_split['order']['dmr'][0])
    n_spon = len(ne_split['order']['spon'][0])
    n_match = min([n_dmr, n_spon])

    for i in range(n_match):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))
        plot_matching_ic_3_conditions(ne_split, axes, i)
        plt.tight_layout()
        fig.savefig(re.sub('.jpg', '-{}.jpg'.format(i), figpath))
        plt.close()


def plot_matching_ic(ax1, ic1, ic2, color1, color2, stim1, stim2, marker_size=10, ymax=None):
    """
    stem plot of matching patterns

    Input:
        ax1: axes to plot on
        ic1, ic2: cNE patterns
        color1, color2: color for the 2 patterns
    """
    # get ylimit for the plot
    if not ymax:
        ymax = max(ic1.max(), ic2.max()) * 1.15
    # plot threshold for ne members
    n_neuron = len(ic1)
    thresh = 1 / np.sqrt(n_neuron)
    ax1.plot([0, len(ic1) + 1], [thresh, thresh], 'k--')
    ax1.plot([0, len(ic1) + 1], [0, 0], 'k')
    ax1.plot([0, len(ic1) + 1], [-thresh, -thresh], 'k--')

    x = range(n_neuron)
    ax2 = ax1.twinx()

    # plot on the left axes
    markerline, stemline, baseline = ax1.stem(
        range(1, n_neuron + 1), ic1,
        markerfmt='o', basefmt='k')
    plt.setp(markerline, markersize=marker_size, color=color1)
    plt.setp(stemline, color=color1)
    ax1.set_xlim([0, n_neuron + 1])
    ax1.set_xticks(range(5, n_neuron + 1, 5))
    ax1.set_ylim([-ymax, ymax])
    ax1.set_yticks([-0.5, 0, 0.5])
    ax1.set_yticklabels([-0.5, 0, 0.5], fontsize=15)
    ax1.spines['left'].set_color(color1)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.tick_params(axis='y', colors=color1)
    ax1.text(8, 0.5, '|corr|={:.2f}'.format(np.corrcoef(ic1, ic2)[0][1]), fontsize=12)
    # plot on the right axes
    markerline, stemline, baseline = ax2.stem(
        range(1, n_neuron + 1), ic2,
        markerfmt='o', basefmt='k')
    plt.setp(markerline, markersize=marker_size, color=color2)
    plt.setp(stemline, color=color2)
    ax2.set_xlim([0, n_neuron + 1])
    ax2.set_ylim([-ymax, ymax])
    ax2.set_yticks([-0.5, 0, 0.5])
    ax2.set_yticklabels([-0.5, 0, 0.5], fontsize=15)
    ax2.invert_yaxis()
    ax2.spines.top.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines['right'].set_color(color2)
    ax2.tick_params(axis='y', colors=color2)

    xbox1 = TextArea(stim1, textprops=dict(color=color1, size=15, ha='center', va='center'))
    xbox2 = TextArea('vs', textprops=dict(color='k', size=15, ha='center', va='center'))
    xbox3 = TextArea(stim2, textprops=dict(color=color2, size=15, ha='center', va='center'))
    xbox = HPacker(children=[xbox1, xbox2, xbox3], align="left", pad=0, sep=5)
    anchored_xbox = AnchoredOffsetbox(loc=8, child=xbox, pad=0., frameon=False, bbox_to_anchor=(0.5, 1.05),
                                      bbox_transform=ax1.transAxes, borderpad=0.)
    ax1.add_artist(anchored_xbox)


def plot_matching_ic_3_conditions(ne_split, axes, idx_match, marker_size=10, ymax=None):
    dmr_first = ne_split['dmr_first']
    colors = colors_split
    if dmr_first:
        order = ne_split['order']['dmr'][1][idx_match]
        ic_dmr = ne_split['dmr1'].patterns[order]
        order = ne_split['order']['spon'][0][idx_match]
        ic_spon = ne_split['spon0'].patterns[order]
    else:
        order = ne_split['order']['dmr'][0][idx_match]
        ic_dmr = ne_split['dmr0'].patterns[order]
        order = ne_split['order']['spon'][1][idx_match]
        ic_spon = ne_split['spon1'].patterns[order]

        plot_matching_ic(axes[0], ic_dmr, ic_spon, colors[0], colors[2], 'dmr1', 'spon2', marker_size)
        order0 = ne_split['order']['dmr'][0][idx_match]
        order1 = ne_split['order']['dmr'][1][idx_match]
        plot_matching_ic(axes[1],
                         ne_split['dmr0'].patterns[order0],
                         ne_split['dmr1'].patterns[order1],
                         colors[0], colors[1], 'dmr1', 'dmr22',
                         marker_size, ymax)
        order0 = ne_split['order']['spon'][0][idx_match]
        order1 = ne_split['order']['spon'][1][idx_match]
        plot_matching_ic(axes[2],
                         ne_split['spon0'].patterns[order0],
                         ne_split['spon1'].patterns[order1],
                         colors[2], colors[3], 'spon1', 'spon2',
                         marker_size, ymax)
        axes[2].set_xlabel('neuron #')


def plot_ne_split_ic_weight_corr(ne_split, ax=None, figpath=None):
    """
    heatmap of correlation values among matching patterns
    """
    corr_mat = ne_split['corr_mat']
    n_dmr = len(ne_split['order']['dmr'][0])
    n_spon = len(ne_split['order']['spon'][0])
    if not ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(corr_mat, aspect='auto', cmap='Greys', vmin=0, vmax=1)
    axins = inset_axes(
        ax,
        width="8%",  # width: 5% of parent_bbox width
        height="80%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = plt.colorbar(im, cax=axins)
    cb.ax.tick_params(axis='y', direction='in')
    axins.set_title('  |corr|', fontsize=18, pad=20)
    cb.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axins.tick_params(axis='both', which='major', labelsize=15)

    # draw boundary for correlation matrix of dmr-evoked activities
    offset = 0.38
    shift = 0.18
    p = mpl.patches.Rectangle((-offset, -offset), n_dmr - shift, n_dmr - shift,
                              facecolor='none', edgecolor='green', linewidth=5)
    ax.add_patch(p)
    # draw boundary for correlation matrix of spontanoues activities
    p = mpl.patches.Rectangle((n_dmr - offset, n_dmr - offset), n_spon - shift, n_spon - shift,
                              facecolor='none', edgecolor='orange', linewidth=5)
    ax.add_patch(p)
    # draw boundary for correlation matrix of corss conditions
    if ne_split['dmr_first']:
        xy = (-offset, n_dmr - offset)
        x, y = n_dmr - shift, n_spon - shift
    else:
        xy = (n_dmr - offset, -offset)
        x, y = n_spon - shift, n_dmr - shift
    p = mpl.patches.Rectangle(xy, x, y,
                              facecolor='none', edgecolor='purple', linewidth=5)
    ax.add_patch(p)
    order = ne_split['order']['dmr'][0] + ne_split['order']['spon'][0]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    order = ne_split['order']['dmr'][1] + ne_split['order']['spon'][1]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    if figpath:
        fig.savefig(figpath)
        plt.close(fig)


def plot_ne_split_ic_weight_null_corr(ne_split, figpath):
    """
    histogram of the distribution of null correlation, significance threshold and real values
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 8))
    c = 0
    for key, corr in ne_split['corr_null'].items():
        thresh = ne_split['corr_thresh'][key]
        ax = axes[c]
        ax.hist(corr, np.linspace(0, 1, 51), color='k',
                weights=np.repeat([1 / len(corr)], len(corr)))
        y = ax.get_ylim()
        ax.plot([thresh, thresh], y, 'r')
        corr_real = ne_split['corr'][key]
        for corr_val in corr_real:
            ax.plot([corr_val, corr_val], y, 'b')
        ax.plot([thresh, thresh], y, 'r')
        ax.set_xlim([0, 1])
        ax.set_ylim(y)
        ax.set_xlabel('|correlation|')
        ax.set_ylabel('ratio')
        ax.set_title(key)
        c += 1
    plt.tight_layout()
    fig.savefig(figpath)
    plt.close(fig)


def figure2(datafolder, figfolder):
    """
    Figure2: stability of cNEs on spantaneous and sensory-evoked activity blocks
    schematics of activity blocks
    correlation matrix of cNE patterns
    example stem plot of matching cNEs
    distribution of significant correlation values
    box plot and scatter plot comparing corss-condition and within condition pattern correlation

    Input:
        datafolder: path to *ne-20dft-dmr.pkl files
        figfolder: path to save figure
    Return:
        None
    """
    # use example recording to plot correlation matrix
    ne_file = os.path.join(datafolder, '200821_015617-site6-5655um-25db-dmr-31min-H31x64-fs20000-ne-20dft-split.pkl')
    with open(ne_file, 'rb') as f:
        ne_split = pickle.load(f)

    fig = plt.figure(figsize=figure_size)
    y_start = 0.79
    x_start = 0.06

    # correlation matrix
    ax = fig.add_axes([x_start, y_start, 0.12, 0.09])
    plot_ne_split_ic_weight_corr(ne_split, ax=ax)
    ybox1 = TextArea("dmr1 ", textprops=dict(color=colors_split[0], size=15, rotation=90, ha='left', va='bottom'))
    ybox2 = TextArea("spon1", textprops=dict(color=colors_split[2], size=15, rotation=90, ha='left', va='bottom'))
    ybox = VPacker(children=[ybox1, ybox2], align="bottom", pad=0, sep=30)
    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.15, 0.12),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_ybox)

    xbox1 = TextArea("dmr2 ", textprops=dict(color=colors_split[1], size=15, ha='center', va='center'))
    xbox2 = TextArea("spon2", textprops=dict(color=colors_split[3], size=15, ha='center', va='center'))
    xbox = HPacker(children=[xbox1, xbox2], align="left", pad=0, sep=30)
    anchored_xbox = AnchoredOffsetbox(loc=8, child=xbox, pad=0., frameon=False, bbox_to_anchor=(0.5, 1.05),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_xbox)

    ax.set_ylabel('cNE #', labelpad=20)
    ax.set_xlabel('cNE #')
    ax.text(4.75, 1.25, 'A', color='w', weight='bold', fontsize=15)
    ax.text(5.75, 2.25, 'B', color='w', weight='bold', fontsize=15)
    # stem plot for matching patterns
    x_start = 0.27
    fig_y = 0.045
    fig_x = 0.1
    space_x = 0.15
    space_y = 0.07
    axes = []
    for i in range(3):
        axes.append(fig.add_axes([x_start, 0.79 + (2 - i) * space_y, fig_x, fig_y]))
    axes[2].set_ylabel('ICweight')
    axes[0].set_title('A', pad=20)
    plot_matching_ic_3_conditions(ne_split, axes, 1, marker_size=7, ymax=0.7)
    axes = []
    for i in range(3):
        axes.append(fig.add_axes([x_start + space_x, 0.79 + (2 - i) * space_y, fig_x, fig_y]))
    plot_matching_ic_3_conditions(ne_split, axes, 2, marker_size=7, ymax=0.7)
    axes[0].set_title('B', pad=20)

    # distribution of significant correlations
    x_start = x_start + 2 * space_x + 0.02
    fig_y = 0.05
    fig_x = 0.12
    df = pd.read_json(os.path.join(re.sub('pkl', 'summary', datafolder), 'split_cNE.json'))
    bins = np.linspace(0.5, 1, 21)
    for i, stim in enumerate(['cross', 'spon', 'dmr']):
        ax = fig.add_axes([x_start, 0.79 + i * space_y, fig_x, fig_y])
        sns.histplot(data=df[(df.stim == stim) & df.corr_sig], x='corr', bins=bins,
                     hue="region", palette=[MGB_color[0], A1_color[0]], hue_order=['MGB', 'A1'],
                     ax=ax, legend=False, stat='proportion', common_norm=False)
        ax.set_title(stim)
        ax.set_ylim([0, 0.16])
        n_ne_sig = np.empty(2)
        n_ne = np.empty(2)
        text_y = 0.13
        for ii, region in enumerate(['MGB', 'A1']):
            corr_sig = df[(df.stim == stim) & (df.region == region)]['corr_sig']
            ratio = corr_sig.mean()
            n_ne_sig[ii] = corr_sig.sum()
            n_ne[ii] = len(corr_sig)
            ax.text(0.5, text_y - 0.022 * ii, '{:.1f}%'.format(ratio * 100),
                    color=eval('{}_color[0]'.format(region)), fontsize=12)
        _, p = chi_square(n_ne_sig, n_ne)
        ax.text(0.5, text_y - 0.022 * 2, 'p = {:.3f}'.format(p), color='k', fontsize=12)

        if i == 0:
            plt.legend(loc='upper right', labels=['A1', 'MGB'], fontsize=12)
            ax.set_xlabel('| correlation |')
            ax.set_ylabel('ratio')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

    # summary plots comparing significant correlation values
    x_start = x_start + fig_x + 0.06
    fig_x = 0.2
    ax = fig.add_axes([x_start, 0.79 + 1.5 * space_y, fig_x, fig_y + 0.5 * space_y])
    df['region_stim'] = df[['region', 'stim']].apply(tuple, axis=1)
    df['region_stim'] = df['region_stim'].apply(lambda x: '_'.join([str(y) for y in x]))
    my_order = df.groupby(by=['region', 'stim'])['corr'].mean().iloc[::-1].index
    my_order = ['_'.join([str(y) for y in x]) for x in my_order]
    order_idx = [1, 4, 0, 3, 2, 5]
    my_order = [my_order[i] for i in order_idx]
    boxplot_scatter(ax, x='region_stim', y='corr', data=df,
                    order=my_order, hue='region_stim', palette=colors_split, hue_order=my_order,
                    jitter=0.3, legend=False, notch=True, size=2, alpha=1,
                    linewidth=1)
    ax.set_xticklabels(['\n'.join(x.split('_')) for x in my_order], fontsize=15)
    ax.set_xlim([-1, 6])
    ax.set_ylabel('|correlation|')
    # significance test between MGB and A1
    for i in range(3):
        _, p = stats.mannwhitneyu(x=df[df['region_stim'] == my_order[i * 2]]['corr'],
                                  y=df[df['region_stim'] == my_order[i * 2 + 1]]['corr'])
        plot_significance_star(ax, p, [i * 2, i * 2 + 1], 1.05, 1.07)
    # significance test between dmr and spon
    for i in range(2):
        _, p = stats.mannwhitneyu(x=df[df['region_stim'] == my_order[i]]['corr'],
                                  y=df[df['region_stim'] == my_order[i + 2]]['corr'])
        plot_significance_star(ax, p, [i, i + 2], 1.05, 1.07)
    # significance test between cross condition and within condition in MGB
    for i in range(2):
        _, p = stats.mannwhitneyu(x=df[df['region_stim'] == my_order[4]]['corr'],
                                  y=df[df['region_stim'] == my_order[2 - i * 2]]['corr'])
        plot_significance_star(ax, p, [2 - i * 2, 4], 1.05 + i * 0.06, 1.04 + i * 0.06)
    # significance test between dmr and cross
    for i in range(2):
        _, res = stats.mannwhitneyu(x=df[df['region_stim'] == my_order[5]]['corr'],
                                    y=df[df['region_stim'] == my_order[3 - i * 2]]['corr'])
        plot_significance_star(ax, p, [3 - i * 2, 5], 1.16 + i * 0.06, 1.15 + i * 0.06)

    ax.set_ylim([0, 1.3])
    ax.set_xlabel('')

    # scatter plots
    fig_x = 0.08
    space_x = 0.12
    df_cross = pd.concat([
        df[(df.stim == 'cross') & (df.dmr_first)].merge(
            df[(df.stim == 'dmr') & (df.dmr_first)],
            left_on=['exp', 'region', 'idx1'], right_on=['exp', 'region', 'idx2'],
            suffixes=(None, '_dmr'), how='left').merge(
            df[(df.stim == 'spon') & (df.dmr_first)],
            left_on=['exp', 'region', 'idx2'], right_on=['exp', 'region', 'idx1'],
            suffixes=(None, '_spon'), how='left'),
        df[(df.stim == 'cross') & (~df.dmr_first)].merge(
            df[(df.stim == 'dmr') & (~df.dmr_first)],
            left_on=['exp', 'region', 'idx2'], right_on=['exp', 'region', 'idx1'],
            suffixes=(None, '_dmr'), how='left').merge(
            df[(df.stim == 'spon') & (~df.dmr_first)],
            left_on=['exp', 'region', 'idx1'], right_on=['exp', 'region', 'idx2'],
            suffixes=(None, '_spon'), how='left')
    ])
    regions = ['MGB', 'A1']
    for i in range(2):
        ax = fig.add_axes([x_start + i * space_x, 0.79, fig_x, fig_y])
        sns.scatterplot(data=df_cross[df_cross.region == regions[i]],
                        x='corr', y='corr_dmr',
                        ax=ax, color=colors_split[0], s=10)
        sns.scatterplot(data=df_cross[df_cross.region == regions[i]],
                        x='corr', y='corr_spon',
                        ax=ax, color=colors_split[2], s=10)
        ax.plot([0, 1], [0, 1], color='k')

        if i == 0:
            ax.set_xlabel('cross condition')
            ax.set_ylabel('within condition')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(regions[i], color=eval('{}_color[0]'.format(regions[i])))

        _, p = stats.wilcoxon(x=df_cross[df_cross.region == regions[i]]['corr'],
                              y=df_cross[df_cross.region == regions[i]]['corr_dmr'])
        ax.text(0.35, 0.1, 'p = {:.1e}'.format(p), color=colors_split[0], fontsize=12)
        _, p = stats.wilcoxon(x=df_cross[df_cross.region == regions[i]]['corr'],
                              y=df_cross[df_cross.region == regions[i]]['corr_spon'])
        ax.text(0.35, 0.25, 'p = {:.1e}'.format(p), color=colors_split[2], fontsize=12)

    fig.savefig(os.path.join(figfolder, 'fig2.png'))
    fig.savefig(os.path.join(figfolder, 'fig2.pdf'))


# ----------------------plot formatting tools -------------------------------------------
def boxplot_scatter(ax, x, y, data, order, hue, palette, hue_order,
                    jitter=0.4, legend=False, notch=True, size=5, alpha=1,
                    linewidth=2):
    """box plot on top of strip plot: show distribution and individual data points"""
    sns.stripplot(ax=ax, x=x, y=y, data=data, order=order,
                  hue=hue, palette=palette, hue_order=hue_order, size=size, alpha=alpha,
                  jitter=jitter, legend=legend)
    bplot = sns.boxplot(ax=ax, x=x, y=y, data=data,
                        order=order, notch=notch, flierprops={'marker': ''},
                        linewidth=linewidth, width=jitter * 2)
    for i, box_col in enumerate(palette):

        mybox = bplot.patches[i]
        mybox.set_edgecolor(box_col)
        mybox.set_facecolor('w')

        for j in range(i * 6, i * 6 + 6):
            line = bplot.lines[j]
            line.set_color(box_col)
            line.set_mfc(box_col)
            line.set_mec(box_col)


def plot_significance_star(ax, p, x_bar, y_bar, y_star):
    """add stars for significance tests"""
    if p < 0.05:
        ax.plot(x_bar, [y_bar, y_bar], 'k')
        if p < 1e-3:
            ax.text((x_bar[0] + x_bar[1]) / 2, y_star, '***',
                    horizontalalignment='center', verticalalignment='center')
        elif p < 1e-2:
            ax.text((x_bar[0] + x_bar[1]) / 2, y_star, '**',
                    horizontalalignment='center', verticalalignment='center')
        else:
            ax.text((x_bar[0] + x_bar[1]) / 2, y_star, '*',
                    horizontalalignment='center', verticalalignment='center')


# ------------------------------- ne properties 2 -----------------------------------------------------------------
def plot_ne_stim_response_type(datafolder, savefolder, rf='strf'):
    stim_sig = pd.DataFrame({'exp': [], 'ne': [], 'probe': [], 'ne_sig': [], 'member_sig': []})
    files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)

        session_file_path = re.sub('-ne-.*.pkl', r'.pkl', ne.file_path)
        with open(session_file_path, 'rb') as f:
            session = pickle.load(f)

        for idx, members in ne.ne_members.items():

            if rf == 'strf':
                ne_sig = ne.ne_units[idx].strf_sig
                member_sig = []
                for member in members:
                    member_sig.append(session.units[member].strf_sig)
            elif rf == 'crh':
                ne_sig = ne.ne_units[idx].crh_sig
                member_sig = []
                for member in members:
                    member_sig.append(session.units[member].crh_sig)
            elif rf == 'stim':
                ne_sig = ne.ne_units[idx].strf_sig or ne.ne_units[idx].crh_sig
                member_sig = []
                for member in members:
                    member_sig.append(session.units[member].strf_sig or session.units[member].crh_sig)

            member_sig = np.array(member_sig).mean()
            stim_sig = pd.concat((stim_sig,
                                  pd.DataFrame({'exp': session.exp, 'ne': idx, 'probe': session.probe, 'ne_sig': ne_sig,
                                                'member_sig': member_sig},
                                               index=[0])), ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # plot back to back histogram
    pie_colors = [colors[x] for x in [1, 7, 5, 3]]
    for i, probe in enumerate(('H31x64', 'H22x32')):
        ax = axes[i, 0]
        rect = mpl.patches.Rectangle((-100, 0.5), 100, 0.5, edgecolor=None, facecolor=colors[0])
        ax.add_patch(rect)
        rect = mpl.patches.Rectangle((0, 0.5), 100, 0.5, edgecolor=None, facecolor=colors[6])
        ax.add_patch(rect)
        rect = mpl.patches.Rectangle((-100, 0), 100, 0.5, edgecolor=None, facecolor=colors[2])
        ax.add_patch(rect)
        rect = mpl.patches.Rectangle((0, 0), 100, 0.5, edgecolor=None, facecolor=colors[4])
        ax.add_patch(rect)

        data = stim_sig[stim_sig.probe == probe]
        h1, _, _ = ax.hist(data[data.ne_sig > 0]['member_sig'], bins=np.linspace(0, 1, 11),
                           orientation='horizontal', color=colors[7], edgecolor='k')
        ax2 = ax.twiny()
        h2, _, _ = ax2.hist(data[data.ne_sig == 0]['member_sig'], bins=np.linspace(0, 1, 11),
                            orientation='horizontal', color=colors[1], edgecolor='k')
        ax2.invert_xaxis()
        ax2.set_xticks([])
        m1 = (np.ceil(max(h1) / 5) + 1) * 5
        m2 = (np.ceil(max(h2) / 5) + 1) * 5
        ax2.set_xlim([m2, -m1])
        ax.set_xlim([-m2, m1])
        ax.plot([-m2, m1], [0.5, 0.5], 'k')
        ax.set_ylim([0, 1])

        y = [len(data[(data.ne_sig == 0) & (data.member_sig >= 0.5)]),
             len(data[(data.ne_sig > 0) & (data.member_sig >= 0.5)]),
             len(data[(data.ne_sig > 0) & (data.member_sig < 0.5)]),
             len(data[(data.ne_sig == 0) & (data.member_sig < 0.5)])
             ]
        ax.set_ylabel('percent of {}+'.format(rf))
        ax = axes[i, 1]
        ax.pie(y, colors=pie_colors, normalize=True,
               wedgeprops={"edgecolor": "k", 'linewidth': 2},
               autopct='%1.1f%%', pctdistance=1.3)
        if i == 0:
            ax.legend(['None-constructive', 'Facilitative', 'Constructive', 'Stimulus-independent'],
                      bbox_to_anchor=(0.8, 1.4))
    fig.savefig(os.path.join(savefolder, 'ne_types_{}.jpg'.format(rf)))


def plot_ne_neuron_stim_response_hist(datafolder, savefolder):
    files = get_A1_MGB_files(datafolder, 'dmr')
    ne_sig = {'A1': [], 'MGB': []}
    neuron_sig = {'A1': [], 'MGB': []}
    for region, filepaths in files.items():
        for file in filepaths:
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            session_file = re.sub('-ne-20dft-dmr', '', file)

            for cne in ne.ne_units:
                ne_sig[region].append(np.array([cne.strf_sig, cne.crh_sig]))

            with open(session_file, 'rb') as f:
                session = pickle.load(f)

            for unit in session.units:
                neuron_sig[region].append(np.array([unit.strf_sig, unit.crh_sig]))
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    c = 0
    for region in files.keys():
        ne_sig[region] = np.array(ne_sig[region])
        neuron_sig[region] = np.array(neuron_sig[region])
        stim_sig = []
        for units in (ne_sig[region], neuron_sig[region]):
            stim_sig.append(np.array([sum(~units[:, 0] & ~units[:, 1]),
                                      sum(units[:, 0] & ~units[:, 1]),
                                      sum(~units[:, 0] & units[:, 1]),
                                      sum(units[:, 0] & units[:, 1])
                                      ]))
        x = np.array(range(4))
        axes[c].bar(x - 0.2, stim_sig[0] / np.sum(stim_sig[0]), 0.4, label='cNEs')
        axes[c].bar(x + 0.2, stim_sig[1] / np.sum(stim_sig[1]), 0.4, label='neurons')
        if c == 0:
            axes[c].legend()
            axes[c].set_xticks([])
        else:
            axes[c].set_xticks(range(4))
            axes[c].set_xticklabels(['-\n-', '+\n-', '-\n+', '+\n+'])
        axes[c].set_ylabel('ratio')
        axes[c].set_title(region)
        c += 1
    fig.savefig(os.path.join(savefolder, 'hist_ne_neuron_stim_encoding.png'))


def plot_ne_neuron_strf_sig_hist(datafolder, savefolder):
    files = get_A1_MGB_files(datafolder, 'dmr')
    ne_sig = {'A1': [], 'MGB': []}
    neuron_sig = {'A1': [], 'MGB': []}
    for region, filepaths in files.items():
        for file in filepaths:
            with open(file, 'rb') as f:
                ne = pickle.load(f)
            session_file = re.sub('-ne-20dft-dmr', '', file)

            for cne in ne.ne_units:
                ne_sig[region].append(cne.strf_sig)

            with open(session_file, 'rb') as f:
                session = pickle.load(f)

            for unit in session.units:
                neuron_sig[region].append(unit.strf_sig)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    c = 0
    for region in ['MGB', 'A1']:
        ne_sig[region] = np.array(ne_sig[region])
        neuron_sig[region] = np.array(neuron_sig[region])
        stim_sig = []
        for units in (ne_sig[region], neuron_sig[region]):
            stim_sig.append(np.array([sum(units), sum(~units)]))
        x = np.array(range(2))
        axes[c].bar(x - 0.2, stim_sig[0] / np.sum(stim_sig[0]), 0.4, label='cNEs')
        axes[c].bar(x + 0.2, stim_sig[1] / np.sum(stim_sig[1]), 0.4, label='neurons')
        axes[c].set_xticks(range(2))
        if c == 0:
            axes[c].legend()
            axes[c].set_xticklabels(['', ''])
        else:
            axes[c].set_xticklabels(['strf +', 'strf -'])
        axes[c].set_ylabel('ratio')
        axes[c].set_title(region)
        c += 1
    plt.tight_layout()
    fig.savefig(os.path.join(savefolder, 'hist_ne_neuron_strf_encoding.png'))


def plot_neuron_ne_subsample_strf_crh(subsample_ri, taxis, faxis, tmfaxis, smfaxis, 
                                      savefolder=r'E:\Congcong\Documents\data\comparison\figure\cNE-subsample-strf_crh'):
    """
    plot sub sampled strf and crh for each member neuron
    """
    nrows = 8
    fig, axes = plt.subplots(nrows=nrows, ncols=6, figsize = figure_size)
    r = 0
    nfig=1
    for i in subsample_ri.index:
        unit = subsample_ri.iloc[i]
        # plot strf
        unit_types = ['neuron', 'ne_spike', 'cNE']
        m = 0
        for c, unit_type in enumerate(unit_types):
            axes[r][c].clear()
            strf = np.array(unit['strf_{}'.format(unit_type)])[:,:,0]
            m = max(np.abs(strf).max(), m)
            plot_strf(ax=axes[r][c], 
                      strf=strf, 
                      taxis=taxis, faxis=faxis)
            axes[r][c].set_title('{:.2f} / {:.1f}'\
                                 .format(np.array(unit['strf_ri_{}'.format(unit_type)]).mean(), 
                                         unit['strf_ptd_{}'.format(unit_type)]))
            if r != nrows - 1:
                axes[r][c].set_xlabel('')
            if c > 0:
                axes[r][c].set_ylabel('')
                axes[r][c].set_yticklabels('')
        for c in range(3):
            axes[r][c].clim(-m, m)
        
        # plot crh
        for j, unit_type in enumerate(unit_types):
            c = j+3
            axes[r][c].clear()
            plot_crh(ax=axes[r][c], 
                     crh=np.array(unit['crh_{}'.format(unit_type)])[:,:,0], 
                     tmfaxis=tmfaxis, smfaxis=smfaxis)
            axes[r][c].set_title('{:.2f}'\
                                 .format(np.array(unit['crh_ri_{}'.format(unit_type)]).mean()))
            if r != nrows - 1:
                axes[r][c].set_xlabel('')
            if c > 3:
                axes[r][c].set_ylabel('')
                axes[r][c].set_yticklabels('')
        
        if r == nrows - 1:
            plt.tight_layout()
            plt.savefig(os.path.join(savefolder, 'subsampled_strf_ri-{}.jpg'.format(nfig)))
            plt.savefig(os.path.join(savefolder, 'subsampled_strf_ri-{}.pdf'.format(nfig)))
            r = 0
            nfig += 1
        else:
            r += 1
        


def plot_ne_member_strf_crh_nonlinearity(ne, figpath):

    # load session file
    session = ne.get_session_data()
    n_neuron = ne.spktrain.shape[0]
    member_thresh = 1 / np.sqrt(n_neuron)  # threshol for membership

    for i, cne in enumerate(ne.ne_units):
        
        idx_ne = cne.unit
        members = ne.ne_members[idx_ne]
        nrows = max(len(members)+1, 5)
        fig, axes = plt.subplots(nrows, 6, figsize=figure_size[0])
    
        for idx in range(3):
            axes[0][idx].remove()
    
        taxis = cne.strf_taxis
        faxis = cne.strf_faxis
        tmfaxis = cne.tmfaxis
        smfaxis = cne.smfaxis
        # plot cNE strf
        plot_strf(axes[0][3], cne.strf, taxis=taxis, faxis=faxis)
        axes[0][3].set_title('{:.2f} / {:.2f}'.format(np.mean(cne.strf_ri), cne.strf_ptd))

        # plot cNE crh
        plot_crh(axes[0][4], cne.crh, tmfaxis=tmfaxis, smfaxis=smfaxis)
        axes[0][4].set_title('{:.2f} / {:.2f}'.format(np.mean(cne.crh_ri), cne.crh_morani))

        # plot cNE nonlinearity
        plot_nonlinearity(axes[0][5], centers=cne.nonlin_centers, fr=cne.nonlin_fr, fr_mean=cne.nonlin_fr_mean)
        axes[0][5].set_title('{:.2f} / {:.2f}'.format(cne.nonlin_asi, np.mean(cne.strf_info)))
        for c in range(3, 6):
           axes[0][c].set_xlabel('')
           axes[0][c].set_ylabel('')
           axes[0][c].set_yticklabels([])
           axes[0][c].set_xticklabels([])

        # plot member strf
        
        for idx, member_idx in enumerate(members):
            unit = session.units[member_idx]
            member = ne.member_ne_spikes[idx_ne][idx]
            nrow = idx + 1
            
            # plot neuron strf
            plot_strf(axes[nrow][0], unit.strf, taxis=taxis, faxis=faxis)
            axes[nrow][0].set_title('{:.2f} / {:.2f}'.format(np.mean(unit.strf_ri), unit.strf_ptd))
            # plot neuron crh
            plot_crh(axes[nrow][1], unit.crh, tmfaxis=tmfaxis, smfaxis=smfaxis)
            axes[nrow][1].set_title('{:.2f} / {:.2f}'.format(np.mean(unit.crh_ri), unit.crh_morani))
            # plot neuron nonlinearity
            plot_nonlinearity(axes[nrow][2], centers=unit.nonlin_centers, fr=unit.nonlin_fr, fr_mean=unit.nonlin_fr_mean)
            axes[nrow][2].set_title('{:.2f} / {:.2f}'.format(unit.nonlin_asi, np.mean(unit.strf_info)))
            
            # plot neuron strf
            plot_strf(axes[nrow][3], member.strf, taxis=taxis, faxis=faxis)
            axes[nrow][3].set_title('{:.2f} / {:.2f}'.format(np.mean(member.strf_ri), member.strf_ptd))
            # plot neuron crh
            plot_crh(axes[nrow][4], member.crh, tmfaxis=tmfaxis, smfaxis=smfaxis)
            axes[nrow][4].set_title('{:.2f} / {:.2f}'.format(np.mean(member.crh_ri), member.crh_morani))
            # plot neuron nonlinearity
            plot_nonlinearity(axes[nrow][5], centers=member.nonlin_centers, fr=member.nonlin_fr, fr_mean=member.nonlin_fr_mean)
            axes[nrow][5].set_title('{:.2f} / {:.2f}'.format(member.nonlin_asi, np.mean(member.strf_info)))
            
            if nrow < len(members):
                for c in range(6):
                    axes[nrow][c].set_xlabel('')
                    axes[nrow][c].set_ylabel('')
                    axes[nrow][c].set_yticklabels([])
                    axes[nrow][c].set_xticklabels([])

                    
            
        plt.tight_layout()
        for r in range(len(members)+1, 5):
            for c in range(6):
                axes[r][c].remove()
        name_base = re.findall('\d{6}_\d{6}.*', ne.file_path)
        name_base = re.sub('.pkl', '-cNE_{}.jpg'.format(idx_ne), name_base[0])
        if session.probe == 'H31x64':
            region = 'MGB'
        else:
            region = 'A1'
        fig.savefig(os.path.join(figpath, region, name_base), dpi=300)
        plt.close(fig)
        

def plot_ne_member_strf_crh_nonlinearity_subsample(ne, taxis, faxis, tmfaxis, smfaxis, figpath):
    
    cnes = ne.cNE.unique()
    # load session file

    for idx_ne, cne in enumerate(cnes):
        
        curr_ne = ne[ne.cNE == cne]
        nrows = max(len(curr_ne), 9)
        fig, axes = plt.subplots(nrows, 9, figsize=np.array(figure_size[0])*2)
        
        nrow = 0
        for idx in curr_ne.index:
            unit = curr_ne.loc[idx]
            
            # plot strf
            for i, unit_type in enumerate(('neuron', 'ne_spike', 'cNE')):
                strf = np.array(eval('unit.strf_{}'.format(unit_type)))[:,:,0]
                plot_strf(axes[nrow][i], strf, taxis, faxis)
                
                # infromation about strf: RI/PTD
                strf_ri = np.nanmean(np.array(eval('unit.strf_ri_{}'.format(unit_type)), dtype=np.float64))
                strf_ptd = np.array(eval('unit.ptd_{}'.format(unit_type))).mean()
                axes[nrow][i].text(1, 50, 'RI:{:.2f}'.format(strf_ri))
                axes[nrow][i].text(1, 40, 'PTD:{:.2f}'.format(strf_ptd))
                
                # title
                if nrow == 0:
                    axes[nrow][i].set_title(unit_type.replace('_', ' '))
                
                # axis labels
                axes[nrow][i].set_ylabel(None)
                axes[nrow][i].set_yticklabels([])
                axes[nrow][i].set_xlabel(None)
                axes[nrow][i].set_xticklabels([])
                    
            # plot crh
            for i, unit_type in enumerate(('neuron', 'ne_spike', 'cNE')):
                i += 3
                crh = np.array(eval('unit.crh_{}'.format(unit_type)))[:,:,0]
                plot_crh(axes[nrow][i], crh, tmfaxis, smfaxis)
                
                # infromation about strf: RI/PTD
                crh_ri = np.nanmean(np.array(eval('unit.crh_ri_{}'.format(unit_type)), dtype=np.float64))
                crh_morani = np.array(eval('unit.morani_{}'.format(unit_type))).mean()
                axes[nrow][i].text(1, 13, 'RI:{:.2f}'.format(crh_ri), color='w')
                axes[nrow][i].text(1, 10, '{:.2f}'.format(crh_morani), color='w')
                
                # title
                if nrow == 0:
                    axes[nrow][i].set_title(unit_type.replace('_', ' '))
                
                # axis labels
                axes[nrow][i].set_ylabel(None)
                axes[nrow][i].set_yticklabels([])
                axes[nrow][i].set_xlabel(None)
                axes[nrow][i].set_xticklabels([])
            
            # plot nonlinearity
            for i, unit_type in enumerate(('neuron', 'ne_spike', 'cNE')):
                i += 6
                axes[nrow][i].clear()
                centers = eval('unit.nonlin_centers_{}'.format(unit_type))
                fr = eval('unit.nonlin_fr_{}'.format(unit_type))[-1]
                # fr_mean = eval('unit.nonlin_fr_mean_{}'.format(unit_type))[-1]
                plot_nonlinearity(axes[nrow][i], centers, fr, 0)
                
                # infromation about strf: RI/PTD
                asi = np.array(eval('unit.asi_{}'.format(unit_type))).mean()
                mi = np.array(eval('unit.mi_{}'.format(unit_type))).mean()
                _, ymax = axes[nrow][i].get_ylim()
                xmin, _ = axes[nrow][i].get_xlim()
                axes[nrow][i].text(xmin+1, ymax * 0.8, 'ASI:{:.2f}'.format(asi))
                axes[nrow][i].text(xmin+1, ymax * 0.6, 'MI:{:.2f}'.format(mi))
                
                # title
                if nrow == 0:
                    axes[nrow][i].set_title(unit_type.replace('_', ' '))
                
                # axis labels


                axes[nrow][i].set_ylabel(None)
                axes[nrow][i].set_yticklabels([])
                axes[nrow][i].set_xlabel(None)
                axes[nrow][i].set_xticklabels([])
            nrow += 1

                    
            
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        for r in range(len(curr_ne), 9):
            for c in range(9):
                axes[r][c].remove()
        fig.savefig(os.path.join(figpath, '{}_{}-{}'.format(ne.exp[0], ne.probe[0], idx_ne)), dpi=300)
        plt.close(fig)











