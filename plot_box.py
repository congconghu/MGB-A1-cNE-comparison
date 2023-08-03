# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:47:51 2022

@author: Congcong
"""
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np
from statsmodels.stats import proportion
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
from scipy.ndimage import convolve
import ne_toolbox as netools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from helper import get_A1_MGB_files, get_distance, chi_square

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'Arial'

mpl.rcParams['axes.linewidth'] = .6
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.major.pad'] = 1.5
mpl.rcParams['ytick.major.pad'] = .5
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6

mpl.rcParams['lines.linewidth'] = .8
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['patch.linewidth'] = .6

cm = 1 / 2.54  # centimeters in inches
figure_size = [(17.6 * cm, 17 * cm), (11.6 * cm, 17 * cm), (8.5 * cm, 17 * cm)]
activity_alpha = 99.5
colors = sns.color_palette("Paired")
A1_color = (colors[1], colors[0])
MGB_color = (colors[5], colors[4])
colors_split = [colors[i] for i in [7, 6, 3, 2, 9, 8]]
fontsize_figure_axes_label = 8
fontsize_figure_tick_label = 7
fontsize_panel_label = 12
marker_size = 10


# ----------------------plot formatting tools -------------------------------------------
def boxplot_scatter(ax, x, y, data, order, hue, palette, hue_order,
                    jitter=0.4, legend=False, notch=True, size=2, alpha=.5,
                    linewidth=1, scatter=True):
    """box plot on top of strip plot: show distribution and individual data points"""
    if scatter:
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


def plot_significance_star(ax, p, x_bar, y_bar, y_star, linewidth=.8, fontsize=10):
    """add stars for significance tests"""
    if p < 0.05:
        ax.plot(x_bar, [y_bar, y_bar], 'k', linewidth=linewidth)
        if p < 1e-3:
            ax.text((x_bar[0] + x_bar[1]) / 2, y_star, '***',
                    horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
        elif p < 1e-2:
            ax.text((x_bar[0] + x_bar[1]) / 2, y_star, '**',
                    horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
        else:
            ax.text((x_bar[0] + x_bar[1]) / 2, y_star, '*',
                    horizontalalignment='center', verticalalignment='center', fontsize=fontsize)


# -------------------------------------- single unit properties -------------------------------------------------------
def plot_strf_df(units, figfolder, order=None, properties=False, smooth=False):
    """
    plot strf of units based on data saved in pandas.DataFrame
    The units are plotted in the order indicated by the order keyword

    Parameters
    ----------
    units : TYPE
        pandas.DataFrame contaning unit strf information
    figfolder : TYPE
        folder to save plots
    order : TYPE, optional
        can take value: 'strf_ri', 'strf_ri_p', 'strf_ri_z', 'strf_ptd', 'strf_info'
    properties : TYPE, optional
        If properties of strf should be plotted. True or False
    smooth : TYPE, optional
        If strf should be SMOOTH. True or False

    Returns
    -------
    None.

    """
    # rearange plot order
    if order:
        if order == 'strf_ri':
            order_idx = (units[order]
                         .apply(np.mean)
                         .sort_values(ascending=False))
        elif order == 'strf_ri_p':
            order_idx = units[order].sort_values(ascending=True)
        elif order in ('strf_ri_z', 'strf_ptd'):
            order_idx = units[order].sort_values(ascending=False)
        elif order == 'strf_info':
            order_idx = units[order].apply(np.mean).sort_values(ascending=False)
    else:
        order_idx = units.reset_index()['index']

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


def batch_plot_strf_df_probe(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-summary',
                             figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\su-strf'):
    units = pd.read_json(os.path.join(datafolder, 'single_units.json'))
    groups = list(units.groupby(['exp', 'probe']).groups.keys())

    for exp, probe in groups:
        if probe != 'H31x64':
            continue
        units_tmp = units[(units.exp == exp) & (units.probe == probe)]
        plot_strf_df_probe(units_tmp)
        plt.savefig(os.path.join(figfolder, f'{exp}-{probe}.jpg'))
        plt.close()


def plot_strf_df_probe(units):
    # rearange plot order
    # depth = np.concatenate(list(units.position))[1::2]
    # top = min(depth)
    # bottom = max(depth)
    fig, axes = plt.subplots(5, 5, figsize=[10, 10])
    axes = axes.flatten()
    for i in range(len(units)):
        strf = np.array(units.iloc[i].strf)
        plot_strf(axes[i],
                  strf,
                  taxis=np.array(units.iloc[i].strf_taxis),
                  faxis=np.array(units.iloc[i].strf_faxis),
                  latency=np.array(units.iloc[i].latency),
                  bf=np.array(units.iloc[i].bf),
                  smooth=False)
        axes[i].set_title(units.iloc[i].position[-1])
        if i == 24:
            break


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


def plot_strf(ax, strf, taxis, faxis, latency=None, bf=None, smooth=False, vmax=None, tlim=None, flim=None,
              tlabels=np.array([75, 50, 25, 0]), flabels_arr = np.array([0.5, 2, 8, 32])):
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
    if strf.ndim == 3:
        strf = np.sum(strf, axis=0)
        
    if tlim:
        idx_start = np.where(taxis == tlim[0])[0][0] + 1
        idx_end = np.where(taxis == tlim[1])[0][0] + 1
        taxis = taxis[idx_start: idx_end]
        strf = strf[:, idx_start:idx_end]
    if flim:
        idx_start = np.where(faxis >= flim[0]*1e3)[0][0] + 1
        idx_end = np.where(faxis >= flim[1]*1e3)[0][0] + 1
        faxis = faxis[idx_start: idx_end]
        strf = strf[idx_start:idx_end, :]
     
    max_val = abs(strf).max() * 1.01
    if smooth:
        weights = np.array([[1],
                            [2],
                            [1]],
                           dtype=np.float)
        weights = weights / np.sum(weights[:])
        strf = convolve(strf, weights, mode='constant')
    if vmax is None:
        vmax = max_val
    im = ax.imshow(strf, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)

    
    xticks = np.searchsorted(-taxis, -tlabels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(tlabels)
    ax.set_xlabel('time before spike (ms)')

    faxis = faxis / 1000
    flabels = [str(x).replace('.0', '') for x in flabels_arr]
    yticks = np.searchsorted(faxis, flabels_arr)
    ax.set_yticks(yticks)
    ax.set_yticklabels(flabels)
    ax.set_ylabel('frequency (kHz)', labelpad=-2)

    if bf and latency is not None:
        try:
            idx_t = np.where(taxis <= latency)[0][0]
            idx_f = np.where(faxis >= bf / 1000)[0][0]
        except IndexError:
            return im
        ax.plot([0, idx_t], [idx_f, idx_f], 'k--')
        ax.plot([idx_t, idx_t], [0, idx_f], 'k--')
    return im


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
    # ax.plot(centers, fr, 'ko-', ms=3)
    ax.plot(fr, 'ko-', ms=3)
    # ax.plot([centers[0]-1, centers[-1]+1], [fr_mean, fr_mean], 'k--')
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

    fig = plt.figure(figsize=figure_size[0])
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
        nspk, edges = np.histogram(ne_spikes, bins=ne.edges[::(4000 // ne.df)])
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
            ax.set_ylabel('Activity (a.u.)')

        else:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.spines[['bottom']].set_visible(False)
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
        ax.spines[['bottom']].set_visible(False)
        ax.set_xticks([])
        if i == 0:
            ax.plot([t_start, t_start + 200], [0.1, 0.1], color='k', linewidth=5)
            ax.text(t_start + 50, -0.6, '0.2 s')
            ax.set_yticks(list(range(5, n_neuron, 5)) + [n_neuron + 1])
            ax.set_yticklabels(list(range(5, n_neuron, 5)) + ['cNE'])
            ax.set_ylabel('Neuron #')
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

        fig = plt.figure(figsize=figure_size[0])

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


def plot_ICweight(ax, weights, thresh, direction='h', ylim=None, ylabelpad=None, markersize=10):
    """
    stem plot for cNE patterns
    """
    
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
        ax.set_xlabel('Neuron #')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel('ICweight')
    elif direction == 'v':
        p = mpl.patches.Rectangle((-thresh, 0), 2 * thresh, n_neuron + 1, color='gainsboro')
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
        if ylabelpad:
            ax.set_ylabel('Neuron #', labelpad=ylabelpad)
        else:
            ax.set_ylabel('Neuron #')
        if ylim:
            ax.set_xlim(ylim)
        ax.set_xlabel('ICweight')


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
    ax.set_xlabel('Neuron #')
    ax.set_yticks(range(4, n_neuron, 5))
    ax.set_yticklabels(range(5, n_neuron + 1, 5))
    ax.set_ylabel('Neuron #')
    ax.spines[['top', 'right']].set_visible(True)
    return im


def plot_eigen_values(ax, corr_mat, thresh):
    """
    scatter plot of eigen values, with Marcenko-Pastur threshold
    """
    n_neuron = corr_mat.shape[0]
    eigvals, _ = np.linalg.eig(corr_mat)
    eigvals.sort()
    eigvals = eigvals[::-1]
    ax.plot(range(1, n_neuron + 1), eigvals, 'ko', markersize=2)
    ax.plot([0, n_neuron + 1], [thresh, thresh], 'r--')
    ax.set_xticks(range(5, n_neuron + 1, 5))
    ax.set_xlim([0, n_neuron + 1])
    ax.set_xlabel('PC #')
    ax.set_ylabel('eigenvalue')


def plot_ICweigh_imshow(ax, patterns, members):
    """heatmap of patterns, with member neurons highlighted"""
    patterns = np.transpose(patterns)
    max_val = abs(patterns).max() * 1.01
    im = ax.imshow(patterns, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=-max_val, vmax=max_val)
    ax.spines[['top', 'right']].set_visible(True)
    # highlight members
    for idx, member in members.items():
        ax.scatter([idx] * len(member), member, c='aquamarine', s=2)
    ax.set_xticks(range(patterns.shape[1]))
    ax.set_xticklabels(range(1, patterns.shape[1] + 1))
    ax.set_xlabel('IC #')
    ax.set_yticks(range(4, patterns.shape[0], 5))
    ax.set_yticklabels(range(5, patterns.shape[0] + 1, 5))
    ax.set_ylabel('Neuron #')
    return im


def plot_activity(ax, centers, activity, thresh, t_window, ylim):
    """plot activity with threshold"""
    ax.plot(centers, activity, color='k')
    ax.plot(t_window, thresh * np.array([1, 1]), color='r', linewidth=.5)
    ax.text(t_window[1] - 1e3, 100, 'threshold at 99.5%', color='r', fontsize=6)
    ax.set_xlim(t_window)

    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim(ylim)


def plot_raster(ax, units, offset='idx', color='k', new_order=None, linewidth=1):
    """raster plot of activities of member neurons"""
    for idx, unit in enumerate(units):
        if offset == 'idx':
            pass
        elif offset == 'unit':
            idx = unit.unit
        if new_order is not None:
            idx = new_order[idx]
        ax.eventplot(unit.spiketimes, lineoffsets=idx + 1, linelengths=0.8, colors=color, linewidth=linewidth)


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
    corr_all = {}
    for key, data in xcorr:
        region, member = key
        row = 0 if region == 'MGB' else 2
        col = 0 if member == '(w)' else 1
        corr, im = plot_xcorr_imshow(ax[row][col], data)
        if row == 0 and col == 0: ax[row][col].set_yticks([0, 300, 600, 900])
        if row == 0 and col == 1: ax[row][col].set_yticks([0, 1000, 2000, 3000])
        if row == 2 and col == 0: ax[row][col].set_yticks([0, 300, 600, 900])
        if row == 2 and col == 1: ax[row][col].set_yticks([0, 1000, 2000, 3000])

        plot_xcorr_avg(ax[row + 1][col], corr)
        corr_all['_'.join(key)] = corr

    for region in ('MGB', 'A1'):
        row = 0 if region == 'MGB' else 2
        plot_xcorr_sig(ax[row + 1][0], corr_all[region + '_(w)'], corr_all[region + '_(o)'])

    for row in range(4):
        for col in range(2):
            if col > 0:
                ax[row][col].set_ylabel('')

            if row == 3:
                ax[row][col].set_xticks(range(-200, 201, 50))
                ax[row][col].set_xticklabels(["-200", '', '-100', '', 0, '', '100', '', "200"])
                for line in ax[row][col].xaxis.get_minorticklines():
                    line.set_visible(False)
                ax[row][col].set_xlabel('Lag (ms)')
            else:
                ax[row][col].set_xlabel('')

                if row in (0, 2):
                    ax[row][col].set_xlabel('')
                    ax[row][col].set_xticks(range(0, 401, 50))
                    t = ax[row][col].yaxis.get_offset_text()
                    t.set_x(-0.05)
                else:
                    ax[row][col].set_xticks(range(-200, 201, 50))
                ax[row][col].set_xticklabels([''] * 9)

    # titles
    ax[0][0].set_title('MGB (within cNE)', fontsize=7, color=MGB_color[0], fontweight="bold", pad=2)
    ax[0][1].set_title('MGB (outside cNE)', fontsize=7, color=MGB_color[1], fontweight="bold", pad=2)
    ax[2][0].set_title('A1 (within cNE)', fontsize=7, color=A1_color[0], fontweight="bold", pad=2)
    ax[2][1].set_title('A1 (outside cNE)', fontsize=7, color=A1_color[1], fontweight="bold", pad=2)
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
    axins.set_title('z-scored\ncorr.\n', fontsize=6, linespacing=0.8, pad=2)
    cb.ax.set_yticks(range(4))
    cb.ax.set_yticklabels([0, 1, 2, '>=3'])
    axins.tick_params(axis='both', which='major', labelsize=6)
    if savepath:
        fig.savefig(os.path.join(savepath, 'xcorr_member_nonmember.jpg'))
        plt.close(fig)


def plot_xcorr_sig(ax, corr_w, corr_o, method='permutation'):
    if method == 'mannwhiteneyu':
        sig_idx = []
        for i in range(401):
            _, p = stats.mannwhitneyu(corr_w[:, i], corr_o[:, i])
            if p < 0.01: sig_idx.append(i)
    elif method == 'permutation':
        diff = np.zeros([int(1e4), 401])
        diff[0] = np.mean(corr_w, 0) - np.mean(corr_o, 0)
        corr_all = np.concatenate([corr_w, corr_o])
        n_w = corr_w.shape[0]
        rand_idx = np.array(range(corr_all.shape[0]))
        for i in range(1, int(1e4)):
            np.random.shuffle(rand_idx)
            diff[i] = np.mean(corr_all[rand_idx[:n_w], :], 0) - np.mean(corr_all[rand_idx[n_w:], :], 0)
        thresh = np.percentile(diff, 97.5, axis=0)
        sig_idx = np.where(diff[0] > thresh)[0]
    sig_idx = np.split(sig_idx, np.where(np.diff(sig_idx) != 1)[0] + 1)
    for idx in sig_idx:
        if 201 in idx:
            ax.fill_between(idx - 201, -5, 5, alpha=0.5, edgecolor=None, facecolor='red')
            print(idx[0]-201, idx[-1]-201)
            return

def plot_xcorr_imshow(ax, data):
    """Stack cross correlation curves and plot as heatmap"""
    ax.get_yaxis().get_offset_text().set_visible(False)
    ax.spines[['top', 'right']].set_visible(True)
    corr = data['xcorr'].to_numpy()
    corr = np.stack(corr)
    corr = zscore(corr, axis=1)
    idx_peak = corr.argmax(axis=1)
    order = np.argsort(idx_peak)
    corr = corr[order]
    im = ax.imshow(corr, aspect='auto', origin='lower', vmax=3, vmin=0, cmap='viridis')
    ax.set_xticks(range(0, len(corr[0]), 100))
    ax.set_xticklabels(range(-200, 201, 100))
    ax.set_ylabel('Neuron\npair #')
    ax.set_xlabel('lag (ms)')
    return corr, im

def plot_xcorr_avg(ax, corr):
    """plot averaged cross correlations and standard deviation"""
    x = range(-200, 201)
    corr_avg = corr.mean(axis=0)
    corr_std = corr.std(axis=0)
    # shade for SD
    ax.fill_between(x, corr_avg - corr_std, corr_avg + corr_std,
                    alpha=0.5, edgecolor=None, facecolor='grey')
    # mean value
    ax.plot(x, corr_avg, color='k', linewidth=.8)
    ax.plot([-200, 200], [0, 0], 'k--', linewidth=.6)
    ax.plot([0, 0], [-1.5, 4], 'k--', linewidth=.6)
    ax.set_ylim([-1.5, 4])
    ax.set_yticks(range(0, 5, 2))
    ax.set_xlim([-200, 200])
    ax.set_ylabel('z-scored\ncorrelation')


def plot_xcorr_examples(file, figfolder=r'E:\Congcong\Documents\data\comparison\figure\summary'):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    spktrain = session.spktrain_dmr
    maxlag = 40
    spktrain_shift = np.roll(spktrain, -maxlag, axis=1)
    spktrain_shift = spktrain_shift[:, :-2 * maxlag]
    c = 0
    idx_neuron = [0, 2, 4, 7]
    for idx, i in enumerate(idx_neuron):
        for j in idx_neuron[idx + 1:]:
            xcorr = np.correlate(spktrain[i], spktrain_shift[j], mode='valid')
            plt.bar(np.arange(-20, 20.1, .5), xcorr, color='k')
            plt.title(f'{i}-{j}')
            plt.savefig(os.path.join(figfolder, f'{c}.jpg'))
            plt.close()
            c += 1


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
    for i, region in enumerate(('MGB', 'A1')):
        a, b, r, _, _ = stats.linregress(n_neuron[region], n_ne[region])
        print(region, a, b)
        ax.text(30, 2 - i, 'R\u00b2 = {:.2}'.format(r), color=eval(region + '_color[0]'), fontsize=6)
        x = np.linspace(0, 45, num=10)
        ax.plot(x, a * x + b, color=eval(region + '_color[0]'), linewidth=.8)
        sc.append(ax.scatter(n_neuron[region], n_ne[region], color=eval(region + '_color[0]'),
                             edgecolors='black', linewidth=.2, alpha=.5, s=marker_size))
        print(region, np.mean(n_ne[region]), np.std(n_ne[region]))
    ax.legend(sc, ['MGB (n=34)', 'A1 (n=17)'], fontsize=6, handletextpad=0, labelspacing=.1, borderpad=.3,
              fancybox=False, edgecolor='black', loc='upper left')
    legend = ax.get_legend()
    legend.get_frame().set_linewidth(.6)
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('# of cNEs')
    ax.set_xlim([0, 45])
    ax.set_xticks(range(0, 50, 15))
    ax.set_ylim([0, 10])
    ax.set_yticks(range(0, 11, 2))


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
            print(region,
                  np.mean(ne_size_df[ne_size_df['region'] == region]['size']),
                  np.std(ne_size_df[ne_size_df['region'] == region]['size']))
        boxplot_scatter(ax, x='region', y='size', data=ne_size_df,
                        order=('MGB', 'A1'), hue='region',
                        palette=(MGB_color[0], A1_color[0]), hue_order=('MGB', 'A1'))

        ax.set_xticklabels(['MGB\n(n={})'.format(len(ne_size['MGB'])),
                            'A1\n(n={})'.format(len(ne_size['A1']))])
        ax.set_ylabel('Relative cNE size')
        ax.set_xlabel('')

        res = stats.mannwhitneyu(ne_size['MGB'], ne_size['A1'])
        p = res.pvalue
        print('p=', p)
        plot_significance_star(ax, p, [0, 1], 0.55, 0.56)
        ax.set_ylim([0, 0.6])
        ax.text(-.4, 0.55, 'recordings with {}-{} neurons'.format(n_neuron_filter[0], n_neuron_filter[1]), fontsize=6)
    else:
        sc = []
        for i, region in enumerate(('MGB', 'A1')):
            sc.append(ax.scatter(n_neuron[region], ne_size[region], color=eval(region + '_color[0]'),
                                 edgecolors='black', linewidth=.2, alpha=.5, s=marker_size))
            print(region, np.mean(ne_size[region]), np.std(ne_size[region]))
            a, b, r, _, _ = stats.linregress(n_neuron[region], ne_size[region])
            print(region, a, b)
            ax.text(30, 15 - i, 'R\u00b2 = {:.2}'.format(r), color=eval(region + '_color[0]'), fontsize=6)
            x = np.linspace(0, 45, num=10)
            ax.plot(x, a * x + b, color=eval(region + '_color[0]'))
        n_min = np.min(n_neuron['A1'])
        n_max = np.max(n_neuron['MGB'])
        ax.plot([n_min, n_min], [0, 11], 'k--')
        ax.plot([n_max, n_max], [0, 11], 'k--')
        print(n_min, n_max)
        ax.legend(sc, ['MGB (n={})'.format(len(n_neuron['MGB'])), 'A1 (n={})'.format(len(n_neuron['A1']))],
                  fontsize=6, handletextpad=0, labelspacing=.1, borderpad=.3,
                  fancybox=False, edgecolor='black', loc='upper left')
        ax.set_xlabel('# of neurons')
        ax.set_ylabel('{} cNE size'.format(plot_type))
        ax.set_xlim([0, 45])
        ax.set_xticks(range(0, 50, 15))


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
    c = 0
    b = []
    n = []
    obs = []
    for region in ('MGB', 'A1'):
        obs.append(np.histogram(n_participation[region], bins=range(5))[0])
        n.append(obs[-1].sum())
        proportion = obs[-1] / obs[-1].sum()
        b.append(ax.bar(np.arange(4) - 0.25 + c * 0.5, proportion, width=0.4, color=eval(region + '_color[0]')))
        c += 1
    ax.legend(b, ['MGB (n={})'.format(n[0]), 'A1 (n={})'.format(n[1])],
              fontsize=6, handletextpad=.2, labelspacing=.1, frameon=False, bbox_to_anchor=(0.5, .9))
    ax.set_xlabel('# of cNEs a neuron belongs to')
    ax.set_ylabel('Proportion')
    ax.set_ylim([0, 1])
    ax.text(-.5, 0.93, 'recordings with {}-{} neurons'.format(n_neuron_filter[0], n_neuron_filter[1]), fontsize=6)
    obs = np.array(obs)
    stats = importr('stats')
    res = stats.fisher_test(obs)
    print(res)
    print(obs)

def ne_member_distance(ax, datafolder, stim, probe, direction='vertical', df=20, linewidth=1.2):
    files = glob.glob(os.path.join(datafolder, '*-{}dft-{}.pkl'.format(df, stim)))
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
        ax.hist(dist_member, range(0, 800, step), color=color,
                weights=np.repeat([1 / len(dist_member)], len(dist_member)))
        ax.hist(dist_nonmember, range(0, 800, step), color='k',
                weights=np.repeat([1 / len(dist_nonmember)], len(dist_nonmember)),
                fill=False, histtype='step', linewidth=linewidth)
        ax.set_xlabel('Pairwise distance (\u03BCm)')
        ax.set_xlim([0, 800])
        x2 = np.mean(dist_member)
        y_space = .009
        ax.scatter(x2, .12, marker='v', facecolor='white', edgecolor=color, s=12, linewidth=.6)
        x1 = np.median(dist_member)
        ax.scatter(x1, .12, marker='v', color=color, s=12, linewidth=.6)
        x4 = np.mean(dist_nonmember)
        ax.scatter(x4, .11, marker='v', facecolor='white', edgecolor='k', s=12, linewidth=.6)
        x3 = np.median(dist_nonmember)
        ax.scatter(x3, .11, marker='v', color='k', s=12, linewidth=.6)
        print('pairwise distance', x1, x2, x3, x4)
        _, p = stats.mannwhitneyu(dist_member, dist_nonmember)
        print('p=', p)
        ax.set_xticks(np.arange(0, 800, 200))
        ax.text(500, .12, 'p = {:.1e}'.format(p), fontsize=6)

    elif direction == 'horizontal':
        ax.hist(dist_member, 2, color=color, align='left', rwidth=0.8,
                weights=np.repeat([1 / len(dist_member)], len(dist_member)))
        ax.hist(dist_nonmember, 2, color='k', align='left', rwidth=0.8,
                weights=np.repeat([1 / len(dist_nonmember)], len(dist_nonmember)),
                fill=False)
        ax.set_xticks([0, 125])
        ax.set_xticklabels(['same shank', 'across shank'])
    ax.set_ylabel('Proportion')


def ne_member_span(ax, datafolder, stim, probe, df=20, linewidth=1.2):
    files = glob.glob(os.path.join(datafolder, '*-{}dft-{}.pkl'.format(df, stim)))
    random.seed(0)
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
        y = .12
        y_space = .009
    else:
        color = A1_color[0]
        step = 25
        y = .165
        y_space = .012
    ax.hist(span_member, range(0, 1001, step * 2), color=color,
            weights=np.repeat([1 / len(span_member)], len(span_member)))
    ax.hist(span_random, range(0, 1001, step * 2), color='k',
            weights=np.repeat([1 / len(span_random)], len(span_random)),
            fill=False, histtype='step', linewidth=linewidth)
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Span (\u03BCm)')
    x2 = np.mean(span_member)
    ax.scatter(x2, y, marker='v', facecolor='white', edgecolor=color, s=12, linewidth=.6)
    x1 = np.median(span_member)
    ax.scatter(x1, y - y_space, marker='v', color=color, s=12, linewidth=.6)
    x4 = np.mean(span_random)
    ax.scatter(x4, y - 2 * y_space, marker='v', facecolor='white', edgecolor='k', s=12, linewidth=.6)
    x3 = np.median(span_random)
    ax.scatter(x3, y - 3 * y_space, marker='v', color='k', s=12, linewidth=.6)

    print('span', x1, x2, x3, x4)
    _, p = stats.mannwhitneyu(span_member, span_random)
    print('p=', p)
    if probe == 'H31x64':
        ax.set_xlim([0, 1300])
        ax.set_xticks(range(0, 1201, 400))
        ax.text(800, y, 'p = {:.1e}'.format(p), fontsize=6)
    else:
        ax.set_ylim([0, .17])
        ax.set_xlim([0, 800])
        ax.set_xticks(range(0, 800, 250))
        ax.text(500, .14, 'p = {:.3f}'.format(p), fontsize=6)


def ne_member_shank_span(ax, datafolder, stim, probe, df=20):
    files = glob.glob(os.path.join(datafolder, '*-{}dft-{}.pkl'.format(df, stim)))
    random.seed(0)
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
    ax.set_ylabel('Proportion')
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
    print('C:')
    n = []
    for i, region in enumerate(('MGB', 'A1')):
        span_ne[region] = np.array(span_ne[region])
        span_all[region] = np.array(span_all[region])
        idx = ~ np.isnan(span_ne[region])
        span_ne[region] = span_ne[region][idx]
        span_all[region] = span_all[region][idx]
        a, b, r, p, _ = stats.linregress(span_all[region], span_ne[region])
        x = np.linspace(0, 6, num=10)
        ax.text(.3, 3.5 - 0.6 * i, 'R\u00b2 = {:.2}'.format(r), color=eval(region + '_color[0]'), fontsize=6)
        print(region, a, b, p)
        ax.plot(x, a * x + b, color=eval(region + '_color[0]'), linewidth=.8)
        sc.append(ax.scatter(span_all[region], span_ne[region], color=eval(region + '_color[0]'),
                             edgecolors='black', linewidth=.2, alpha=.8, s=marker_size))
        n.append(len(span_all[region]))
    ax.legend(sc, ['MGB (n={})'.format(n[0]), 'A1(n={})'.format(n[1])], fontsize=6, handletextpad=0, labelspacing=.1,
              borderpad=.2,
              fancybox=False, edgecolor='black')
    legend = ax.get_legend()
    legend.get_frame().set_linewidth(.6)
    ax.set_xlabel('All neurons')
    ax.set_ylabel('Member neurons')
    ax.set_xlim([0, 6])
    ax.set_ylim([-.3, 6])


def ne_member_freq_span(ax, datafolder, stim, probe, df=20, linewidth=1.2):
    files = glob.glob(os.path.join(datafolder, '*-{}dft-{}.pkl'.format(df, stim)))
    random.seed(0)
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
            fill=False, histtype='step', linewidth=linewidth)
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Frequency span (oct)')
    ax.set_xlim([0, 6])
    ax.set_ylim([0, .4])
    ax.set_yticks(np.arange(0, .41, .1))
    ax.set_yticks(np.arange(0, .41, .1))
    ax.set_ylabel('Proportion')
    x2 = np.mean(span_member)
    y = .36
    y_space = .025
    ax.scatter(x2, y, marker='v', facecolor='white', edgecolor=color, s=12, linewidth=.6)
    x1 = np.median(span_member)
    ax.scatter(x1, y - y_space, marker='v', color=color, s=12, linewidth=.6)
    x4 = np.mean(span_random)
    ax.scatter(x4, y - 2 * y_space, marker='v', facecolor='white', edgecolor='k', s=12, linewidth=.6)
    x3 = np.median(span_random)
    ax.scatter(x3, y - 3 * y_space, marker='v', color='k', s=12, linewidth=.6)
    print('pairwise frequenct distance', x1, x2, x3, x4)
    _, p = stats.mannwhitneyu(span_member, span_random)
    print('p=', p)
    if p < .001:
        ax.text(4, .3, 'p = {:.1e}'.format(p), fontsize=6)
    else:
        ax.text(4, .3, 'p = {:.3f}'.format(p), fontsize=6)


def ne_member_freq_distance(ax, datafolder, stim, probe, df=20, linewidth=1.2):
    files = glob.glob(os.path.join(datafolder, '*-{}dft-{}.pkl'.format(df, stim)))
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

    ax.hist(dist_member, np.linspace(0, 6, 20), color=color,
            weights=np.repeat([1 / len(dist_member)], len(dist_member)))
    ax.hist(dist_nonmember, np.linspace(0, 6, 20), color='k',
            weights=np.repeat([1 / len(dist_nonmember)], len(dist_nonmember)),
            fill=False, histtype='step', linewidth=linewidth)
    ax.set_xlabel('Pairwise frequency difference (oct)')
    ax.set_xlim([0, 6])
    ax.set_ylim([0, .4])
    ax.set_ylabel('Proportion')
    x2 = np.mean(dist_member)
    y = .36
    y_space = .025
    ax.scatter(x2, y, marker='v', facecolor='white', edgecolor=color, s=12, linewidth=.6)
    x1 = np.median(dist_member)
    ax.scatter(x1, y - y_space, marker='v', color=color, s=12, linewidth=.6)
    x4 = np.mean(dist_nonmember)
    ax.scatter(x4, y - 2 * y_space, marker='v', facecolor='white', edgecolor='k', s=12, linewidth=.6)
    x3 = np.median(dist_nonmember)
    ax.scatter(x3, y - 3 * y_space, marker='v', color='k', s=12, linewidth=.6)
    print('pairwise frequenct distance', x1, x2, x3, x4)
    _, p = stats.mannwhitneyu(dist_member, dist_nonmember)
    print('p=', p)
    if p < .001:
        ax.text(4, .3, 'p = {:.1e}'.format(p), fontsize=6)
    else:
        ax.text(4, .3, 'p = {:.3f}'.format(p), fontsize=6)


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


def plot_matching_ic(ax1, ic1, ic2, color1, color2, stim1, stim2, marker_size=3, ymax=None,
                     yticklabels=[True, True], linewidth=.8):
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
    ax1.plot([0, len(ic1) + 1], [thresh, thresh], 'k--', linewidth=.5)
    ax1.plot([0, len(ic1) + 1], [0, 0], 'k', linewidth=linewidth)
    ax1.plot([0, len(ic1) + 1], [-thresh, -thresh], 'k--', linewidth=.5)
    ax2 = ax1.twinx()

    # plot on the left axes
    markerline, stemline, baseline = ax1.stem(
        range(1, n_neuron + 1), ic1,
        markerfmt='o', basefmt=' ')
    plt.setp(markerline, markersize=marker_size, color=color1, linewidth=.8)
    plt.setp(stemline, color=color1, linewidth=linewidth)
    ax1.set_xlim([0, n_neuron + 1])
    ax1.set_xticks(range(5, n_neuron + 1, 5))
    ax1.set_ylim([-ymax, ymax])
    ax1.set_yticks([-0.5, 0, 0.5])
    ax1.set_yticklabels([-0.5, 0, 0.5], fontsize=fontsize_figure_tick_label)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='y', colors=color1)
    ax1.text(8, 0.7, '|corr|={:.2f}'.format(np.corrcoef(ic1, ic2)[0][1]), fontsize=6)
    # plot on the right axes
    markerline, stemline, baseline = ax2.stem(
        range(1, n_neuron + 1), ic2,
        markerfmt='o', basefmt=' ')
    plt.setp(markerline, markersize=marker_size, color=color2, linewidth=linewidth)
    plt.setp(stemline, color=color2, linewidth=linewidth)
    ax2.set_xlim([0, n_neuron + 1])
    ax2.set_ylim([-ymax, ymax])
    ax2.set_yticks([-0.5, 0, 0.5])
    ax2.set_yticklabels([-0.5, 0, 0.5], fontsize=fontsize_figure_tick_label)
    ax2.invert_yaxis()
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(color2)
    ax2.spines[['left']].set_color(color1)
    ax2.tick_params(axis='y', colors=color2)

    xbox1 = TextArea(stim1, textprops=dict(color=color1, size=8, ha='center', va='center'))
    xbox2 = TextArea('vs', textprops=dict(color='k', size=8, ha='center', va='center'))
    xbox3 = TextArea(stim2, textprops=dict(color=color2, size=8, ha='center', va='center'))
    xbox = HPacker(children=[xbox1, xbox2, xbox3], align="left", pad=0, sep=5)
    anchored_xbox = AnchoredOffsetbox(loc=8, child=xbox, pad=0., frameon=False, bbox_to_anchor=(0.5, 1.2),
                                      bbox_transform=ax1.transAxes, borderpad=0.)
    ax1.add_artist(anchored_xbox)
    if not yticklabels[0]:
        ax1.set_yticklabels([])
    if not yticklabels[1]:
        ax2.set_yticklabels([])
    ax2.spines['bottom'].set_visible(False)


def plot_matching_ic_3_conditions(ne_split, axes, idx_match, marker_size=3, ymax=None,
                                  yticklabels=[True, True]):
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

    plot_matching_ic(axes[0], ic_spon, ic_dmr, colors[0], colors[2], 'spon2', 'dmr1',
                     marker_size, yticklabels=yticklabels)
    order0 = ne_split['order']['spon'][0][idx_match]
    order1 = ne_split['order']['spon'][1][idx_match]
    axes[1].set_xticklabels([])
    plot_matching_ic(axes[1],
                     ne_split['spon1'].patterns[order1],
                     ne_split['spon0'].patterns[order0],
                     colors[0], colors[1], 'spon2', 'spon1',
                     marker_size, ymax, yticklabels=yticklabels)
    order0 = ne_split['order']['dmr'][0][idx_match]
    order1 = ne_split['order']['dmr'][1][idx_match]
    axes[0].set_xticklabels([])
    plot_matching_ic(axes[2],
                     ne_split['dmr1'].patterns[order1],
                     ne_split['dmr0'].patterns[order0],
                     colors[3], colors[2], 'dmr2', 'dmr1',
                     marker_size, ymax, yticklabels=yticklabels)

    axes[2].set_xlabel('Neuron #')


def plot_matching_ic_scatter(ax, ne_split, idx_match, marker_size=3):
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
    thresh = 1 / np.sqrt(len(ic_spon))
    ax.scatter(ic_spon, ic_dmr, s=5, color=colors_split[4])
    ax.text(-.15, .4, '{:.2f}'.format(np.corrcoef(ic_spon, ic_dmr)[0, 1]), color=colors_split[4], fontsize=6)
    order0 = ne_split['order']['spon'][0][idx_match]
    order1 = ne_split['order']['spon'][1][idx_match]
    ax.scatter(ne_split['spon1'].patterns[order1], ne_split['spon0'].patterns[order0], color=colors_split[0], s=5)
    ax.text(-.15, .6, '{:.2f}'.format(np.corrcoef(ne_split['spon1'].patterns[order1],
                                                  ne_split['spon0'].patterns[order0])[0, 1]),
            color=colors_split[0], fontsize=6)
    order0 = ne_split['order']['dmr'][0][idx_match]
    order1 = ne_split['order']['dmr'][1][idx_match]
    ax.scatter(ne_split['dmr1'].patterns[order1], ne_split['dmr0'].patterns[order0], color=colors[2], s=5)
    ax.text(-.15, .5, '{:.2f}'.format(np.corrcoef(ne_split['dmr1'].patterns[order1],
                                                  ne_split['dmr0'].patterns[order0])[0, 1]),
            color=colors_split[2], fontsize=6)
    ax.plot([-.2, .8], [-.2, .8], 'k')
    ax.plot([-.2, .8], [thresh, thresh], 'k--')
    ax.plot([thresh, thresh], [-.2, .8], 'k--')
    ax.set_xticks(np.arange(-.2, .9, .2))
    ax.set_yticks(np.arange(-.2, .9, .2))
    ax.set_xlim([-.2, .8])
    ax.set_ylim([-.2, .8])
    ax.text(-.15, .7, '|corr.|=', fontsize=6)


def plot_ne_split_ic_weight_corr(ne_split, ax=None, figpath=None):
    """
    heatmap of correlation values among matching patterns
    """
    corr_mat = ne_split['corr_mat']
    n_dmr = len(ne_split['order']['dmr'][0])
    n_spon = len(ne_split['order']['spon'][0])
    corr_mat = np.concatenate([corr_mat[n_dmr:, :], corr_mat[0:n_dmr, :]])
    corr_mat = np.concatenate([corr_mat[:, n_dmr:], corr_mat[:, 0:n_dmr]], axis=1)
    if not ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(corr_mat, aspect='auto', cmap='Greys', vmin=0, vmax=1)
    # colorbar
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
    axins.set_title('   |corr|', fontsize=6, pad=10)
    cb.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axins.tick_params(axis='both', which='major', labelsize=6)

    # draw boundary for correlation matrix of dmr-evoked activities
    offset = 0.45
    shift = 0.1
    p = mpl.patches.Rectangle((-offset, -offset), n_spon - shift, n_spon - shift,
                              facecolor='none', edgecolor='orange', linewidth=2)
    ax.add_patch(p)
    # draw boundary for correlation matrix of spontanoues activities
    p = mpl.patches.Rectangle((n_spon - offset, n_spon - offset), n_dmr - shift, n_dmr - shift,
                              facecolor='none', edgecolor='green', linewidth=2)
    ax.add_patch(p)
    # draw boundary for correlation matrix of corss conditions
    if not ne_split['dmr_first']:
        xy = (-offset, n_spon - offset)
        x, y = n_spon - shift, n_dmr - shift
    else:
        xy = (n_spon - offset, -offset)
        x, y = n_dmr - shift, n_spon - shift
    p = mpl.patches.Rectangle(xy, x, y,
                              facecolor='none', edgecolor='purple', linewidth=2)
    ax.add_patch(p)
    order = ne_split['order']['spon'][0] + ne_split['order']['dmr'][0]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(np.array(order) + 1)
    order = ne_split['order']['spon'][1] + ne_split['order']['dmr'][1]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(np.array(order) + 1)
    if figpath:
        fig.savefig(figpath)
        plt.close(fig)
    ax.spines[['top', 'right']].set_visible(True)


def plot_ne_split_ic_weight_null_corr(ne_split, figpath):
    """
    histogram of the distribution of null correlation, significance threshold and real values
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 8))
    c = 0
    for key, corr_null in ne_split['corr_null'].items():
        thresh = ne_split['corr_thresh'][key]
        ax = axes[c]
        corr_real = ne_split['corr'][key]
        plot_ne_split_ic_weight_null_corr_panel(ax, thresh, corr_real, corr_null)
        ax.set_title(key)
        c += 1
    plt.tight_layout()
    fig.savefig(figpath)
    plt.close(fig)


def plot_ne_split_ic_weight_null_corr_panel(ax, thresh, corr_real, corr_null, color=None):
    handles = []
    _, _, handle = ax.hist(corr_null, np.linspace(0, 1, 51), color='k',
                           weights=np.repeat([1 / len(corr_null)], len(corr_null)))
    handles.append(handle)
    y = ax.get_ylim()
    h, = ax.plot([thresh, thresh], y, 'r--', linewidth=.8)
    handles.append(h)
    for i, corr_val in enumerate(corr_real):
        if color is None:
            h, = ax.plot([corr_val, corr_val], y, 'b')
            handles.append(h)
        else:
            h, = ax.plot([corr_val, corr_val], y, color[i])
            handles.append(h)
        ax.set_xlim([0, 1])
        ax.set_ylim(y)
        ax.set_xlabel('|Correlation|')
        ax.set_ylabel('Proportion')
    return handles


def plot_box_split_cNE_properties(ax, jsonfile, probe, property='nmember'):
    data = pd.read_json(jsonfile)
    data = data[data.probe == probe]
    stims = ('spon', 'dmr')
    property_split = pd.DataFrame()
    if property == 'nmember':
        data['property1'] = data['member1'].apply(len)
        data['property2'] = data['member2'].apply(len)
    elif property == 'freq_span':
        data['property1'] = data['freq_span1']
        data['property2'] = data['freq_span2']
    for i in range(2):
        stim = stims[i]
        data_tmp = data[data.stim == stim]
        block1 = data_tmp['property1']  # property for first block
        block2 = data_tmp['property2']  # property for second block

        for j in range(2):
            property_tmp = pd.DataFrame({'property': eval(f'block{j + 1}')})
            property_tmp['stim'] = stim
            property_tmp['order'] = str(j + 1)
            property_split = pd.concat([property_split, property_tmp])
    property_split['stim_order'] = property_split['stim'] + property_split['order']
    order = [x + y for x in stims for y in ('1', '2')]
    property_split.reset_index(inplace=True, drop=True)
    cs = [colors_split[x] for x in [0, 1, 2, 3]]
    boxplot_scatter(ax, x='stim_order', y='property', data=property_split, order=order,
                    palette=cs, hue='stim_order', hue_order=order, jitter=.3)
    if property == 'nmember':
        ax.set_ylabel('# of cNE members')
    ax.set_xlabel('')
    ax.set_xticklabels(['block1', 'block2', 'block1', 'block2'])
    model = ols("property ~ stim + stim:order", data=property_split).fit()
    anova_df = sm.stats.anova_lm(model)
    print(property, probe)
    print(anova_df)


def plot_extra_member_strf_sig(ax, datafolder, probe):
    data = pd.read_json(os.path.join(datafolder, 'split_cNE.json'))
    su = pd.read_json(os.path.join(datafolder, 'single_units.json'))
    data = data[(data.probe == probe) & (data.stim == 'cross')]
    strf_sig = {'share': [], 'dmr': [], 'spon': []}
    for i in range(len(data)):
        ne = data.iloc[i]
        member_share = ne.member_share
        member_spon = set(ne.member1).difference(member_share)
        member_dmr = set(ne.member2).difference(member_share)
        su_exp = su[(su.exp == ne.exp) & (su.probe == ne.probe)]
        if ne.dmr_first:
            member_spon, member_dmr = member_dmr, member_spon
        for member_type in ('share', 'dmr', 'spon'):
            strf_sig_tmp = []
            for member in eval(f'member_{member_type}'):
                strf_sig_tmp.append(su_exp[su_exp['index'] == member].strf_sig.item())
            strf_sig[member_type].extend(strf_sig_tmp)

    strf_sig_prc = {key: np.mean(val) for key, val in strf_sig.items()}
    plt.bar(range(4), [strf_sig_prc['spon'], strf_sig_prc['dmr'], strf_sig_prc['share'], np.mean(su.strf_sig)],
            color=[colors_split[0], colors_split[2], colors_split[4], [.5, .5, .5]]
            )


# ------------------------------- cNE significance-----------------------------------------------------------------
def plot_num_cne_vs_shuffle(ax, datafolder='E:\Congcong\Documents\data\comparison\data-summary'):
    df = pd.read_json(os.path.join(datafolder, 'num_ne_data_vs_shuffle.json'))
    df['stim_region_shuffle'] = df[['stim', 'region', 'shuffled']].apply(tuple, axis=1)
    df['stim_region_shuffle'] = df['stim_region_shuffle'].apply(lambda x: (str(tmp) for tmp in x))
    df['stim_region_shuffle'] = df['stim_region_shuffle'].apply(lambda x: '_'.join(x))
    my_order = np.unique(df['stim_region_shuffle'])
    # order of bar plot:
    # 'spon_MGB_0', 'spon_MGB_1', 'spon_A1_0', 'spon_A1_1', 'dmr_MGB_0', 'dmr_MGB_1', 'dmr_A1_0', 'dmr_A1_1'
    my_order = my_order[[6, 7, 4, 5, 2, 3, 0, 1]]
    my_order = {key: i for i, key in enumerate(my_order)}
    m = df.groupby('stim_region_shuffle')['n_ne'].mean()  # mean number of cNE
    print(m)
    sd = df.groupby('stim_region_shuffle')['n_ne'].std()  # std for errorbar
    print(sd)
    m = m.reset_index()
    sd = sd.reset_index()
    key = m['stim_region_shuffle'].map(my_order)
    m = m.iloc[key.argsort()]
    m = m['n_ne']
    sd = sd.iloc[key.argsort()]
    sd = sd['n_ne']

    m.plot.bar(color=np.concatenate([MGB_color, A1_color, MGB_color, A1_color]), ax=ax)
    ax.errorbar(range(8), m, yerr=sd, fmt='None', color="k", capsize=5, linewidth=1)

    c = 0
    line_color = np.ones(3) * .5
    p_all = []
    for stim in ('dmr', 'spon'):
        for region in ('MGB', 'A1'):
            df_tmp = df[(df['stim'] == stim) & (df['region'] == region)]
            n1 = np.array(df_tmp[df_tmp.shuffled == 1]['n_ne'])
            n2 = np.array(df_tmp[df_tmp.shuffled == 0]['n_ne'])
            # significance test
            _, p = stats.wilcoxon(n1, n2)
            p_all.append(p)
            print(stim, region, len(n1), p, np.mean(n1), np.mean(n2))
            for i in range(len(n1)):
                ax.plot([c, c + 1], [n1[i], n2[i]], color=line_color, linewidth=.6)
            c += 2
    p_corrected = multipletests(p_all, alpha=0.05, method='b')[1]
    print(p_corrected)
    for c in range(0, 7, 2):
        plot_significance_star(ax, p_corrected[c // 2], [c, c + 1], 8.2, 8.5)

    ax.set_xticklabels(['\N{MINUS SIGN}', '+'] * 4, rotation=0)
    ax.set_ylabel('# of cNEs', fontsize=fontsize_figure_axes_label)
    ax.set_yticks(range(0, 10, 2))
    # add label for shuffle
    trans = ax.get_xaxis_transform()
    ax.annotate('shuffle', xy=(-1, -.15), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)
    # add label for region
    ax.annotate('MGB', xy=(.5, -.3), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)
    ax.annotate('A1', xy=(2.5, -.3), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)
    ax.annotate('MGB', xy=(4.5, -.3), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)
    ax.annotate('A1', xy=(6.5, -.3), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)
    # add label for spon and dmr
    ax.plot([-.2, 3.2], [-.4, -.4], color="k", transform=trans, clip_on=False, linewidth=.8)
    ax.plot([4 - .2, 7.2], [-.4, -.4], color="k", transform=trans, clip_on=False, linewidth=.8)
    ax.annotate('spon', xy=(1.5, -.5), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)
    ax.annotate('dmr', xy=(5.5, -.5), xycoords=trans, ha="center", va="center", fontsize=fontsize_figure_tick_label)


def plot_ne_sig_corr_hist(ax, ne_real, ne_shuffled, region, bins, linewidth=1.2):
    data = ne_real[(~ne_real.pattern_corr.apply(np.isnan)) & (ne_real.region == region)]
    sns.histplot(data=data, x="pattern_corr",
                 bins=bins, color=eval('{}_color[0]'.format(region)),
                 element="step", fill=False, stat='probability', ax=ax, linewidth=linewidth)
    n_unit = len(data)
    sns.histplot(data=data[data.corr_sig], x="pattern_corr",
                 bins=bins, color=eval('{}_color[0]'.format(region)),
                 ec=eval('{}_color[0]'.format(region)), fill=True, weights=1 / n_unit, ax=ax, linewidth=linewidth)
    ratio = np.array(data['corr_sig']).mean()
    print(region, ratio, len(np.array(data['corr_sig'])), np.array(data['corr_sig']).sum())
    bootstrap_stats = stats.bootstrap([np.array(data['corr_sig']).astype(int)], np.mean,
                                      confidence_level=0.95, random_state=1, method='percentile')
    print(bootstrap_stats)
    data = ne_shuffled[ne_shuffled.region == region]
    sns.histplot(data=data, x="pattern_corr",
                 bins=bins, color=eval('{}_color[1]'.format(region)),
                 element="step", fill=False, stat='probability', ax=ax, linewidth=linewidth)
    n_unit = len(data)
    sns.histplot(data=data[data.corr_sig], x="pattern_corr",
                 bins=bins, color=eval('{}_color[1]'.format(region)),
                 ec=eval('{}_color[1]'.format(region)), fill=True, weights=1 / n_unit, ax=ax, linewidth=linewidth)
    ratio = np.array(data['corr_sig']).mean()
    print(region, ratio, len(np.array(data['corr_sig'])), np.array(data['corr_sig']).sum())

    bootstrap_stats = stats.bootstrap([np.array(data['corr_sig']).astype(int)], np.mean,
                                      confidence_level=0.95, random_state=1, method='percentile')
    print(bootstrap_stats)
    ax.legend([region, f'{region} shuffled', f'{region} sig', f'{region} shuffled sig'])

    ax.set_xlim([0, 1])
    ax.patch.set_alpha(0)
    ax.set_xlabel('')

    ax.set_ylim([0, 0.2])
    ax.set_ylabel('Proportion', fontsize=fontsize_figure_axes_label)


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
    fig, axes = plt.subplots(nrows=nrows, ncols=6, figsize=figure_size)
    r = 0
    nfig = 1
    for i in subsample_ri.index:
        unit = subsample_ri.iloc[i]
        # plot strf
        unit_types = ['neuron', 'ne_spike', 'cNE']
        m = 0
        for c, unit_type in enumerate(unit_types):
            axes[r][c].clear()
            strf = np.array(unit['strf_{}'.format(unit_type)])[:, :, 0]
            m = max(np.abs(strf).max(), m)
            plot_strf(ax=axes[r][c],
                      strf=strf,
                      taxis=taxis, faxis=faxis)
            axes[r][c].set_title('{:.2f} / {:.1f}' \
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
            c = j + 3
            axes[r][c].clear()
            plot_crh(ax=axes[r][c],
                     crh=np.array(unit['crh_{}'.format(unit_type)])[:, :, 0],
                     tmfaxis=tmfaxis, smfaxis=smfaxis)
            axes[r][c].set_title('{:.2f}' \
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
        nrows = max(len(members) + 1, 5)
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
            plot_nonlinearity(axes[nrow][2], centers=unit.nonlin_centers, fr=unit.nonlin_fr,
                              fr_mean=unit.nonlin_fr_mean)
            axes[nrow][2].set_title('{:.2f} / {:.2f}'.format(unit.nonlin_asi, np.mean(unit.strf_info)))

            # plot neuron strf
            plot_strf(axes[nrow][3], member.strf, taxis=taxis, faxis=faxis)
            axes[nrow][3].set_title('{:.2f} / {:.2f}'.format(np.mean(member.strf_ri), member.strf_ptd))
            # plot neuron crh
            plot_crh(axes[nrow][4], member.crh, tmfaxis=tmfaxis, smfaxis=smfaxis)
            axes[nrow][4].set_title('{:.2f} / {:.2f}'.format(np.mean(member.crh_ri), member.crh_morani))
            # plot neuron nonlinearity
            plot_nonlinearity(axes[nrow][5], centers=member.nonlin_centers, fr=member.nonlin_fr,
                              fr_mean=member.nonlin_fr_mean)
            axes[nrow][5].set_title('{:.2f} / {:.2f}'.format(member.nonlin_asi, np.mean(member.strf_info)))

            if nrow < len(members):
                for c in range(6):
                    axes[nrow][c].set_xlabel('')
                    axes[nrow][c].set_ylabel('')
                    axes[nrow][c].set_yticklabels([])
                    axes[nrow][c].set_xticklabels([])

        plt.tight_layout()
        for r in range(len(members) + 1, 5):
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
        fig, axes = plt.subplots(nrows, 9, figsize=np.array(figure_size[0]) * 2)

        nrow = 0
        for idx in curr_ne.index:
            unit = curr_ne.loc[idx]

            # plot strf
            m = np.max(np.abs([unit.strf_neuron, unit.strf_ne_spike, unit.strf_cNE]))
            for i, unit_type in enumerate(('neuron', 'ne_spike', 'cNE')):
                strf = np.array(eval('unit.strf_{}'.format(unit_type)))[:, :, 5]
                plot_strf(axes[nrow][i], strf, taxis, faxis, vmax=m)

                # infromation about strf: RI/PTD
                strf_ri = np.nanmean(np.array(eval('unit.strf_ri_{}'.format(unit_type)), dtype=np.float64))
                strf_ptd = np.array(eval('unit.ptd_{}'.format(unit_type))).mean()
                strf_mi = np.array(eval('unit.mi_{}'.format(unit_type))).mean()
                axes[nrow][i].text(1, 50, 'RI:{:.2f}'.format(strf_ri))
                axes[nrow][i].text(1, 40, 'PTD:{:.2f}'.format(strf_ptd))
                axes[nrow][i].text(1, 30, 'PTD:{:.3f}'.format(strf_mi))

                # title
                if nrow == 0:
                    axes[nrow][i].set_title(unit_type.replace('_', ' '))

                # axis labels
                axes[nrow][i].set_ylabel(None)
                axes[nrow][i].set_yticklabels([])
                axes[nrow][i].set_xlabel(None)
                axes[nrow][i].set_xticklabels([])
                if i == 0:
                    axes[nrow][i].set_title(unit.member, pad=0)

            # plot crh
            for i, unit_type in enumerate(('neuron', 'ne_spike', 'cNE')):
                i += 3
                crh = np.array(eval('unit.crh_{}'.format(unit_type)))[:, :, 0]
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
                # asi = np.array(eval('unit.asi_{}'.format(unit_type))).mean()
                # mi = np.array(eval('unit.mi_{}'.format(unit_type))).mean()
                _, ymax = axes[nrow][i].get_ylim()
                xmin, _ = axes[nrow][i].get_xlim()
                # axes[nrow][i].text(xmin+1, ymax * 0.8, 'ASI:{:.2f}'.format(asi))
                # axes[nrow][i].text(xmin+1, ymax * 0.6, 'MI:{:.2f}'.format(mi))

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
        fig.savefig(os.path.join(figpath, '{}_{}-{}'.format(ne.exp[0], ne.probe[0], cne)), dpi=300)
        plt.close(fig)


def plot_icweight_match_binsize(ic_matched, figpath):
    for idx_ne in range(ic_matched['patterns'].shape[0]):
        fig = plt.figure(figsize=figure_size[0])
        plot_icweight_match_binsize_fig(
            fig, ic_matched['patterns'][idx_ne, :, :], ic_matched['pearsonr'][idx_ne, :], ic_matched['df'])

        fig.savefig('{}-{}.jpg'.format(figpath, idx_ne), dpi=300)
        plt.close()


def plot_icweight_match_binsize_fig(fig, patterns, corr, dfs, ylabel=True,
                                    start_x=.05, start_y=.1, fig_y=.8, fig_x=.1, space_x=.02):
    dfs = np.array(dfs) / 2
    dfs = [int(x) for x in dfs]
    thresh = 1 / np.sqrt(patterns.shape[1])
    ymax = patterns.max() + .1
    ymin = patterns.min() - .1
    axes = []
    for i, df in enumerate(dfs):
        pattern = patterns[i]

        ax = fig.add_axes([start_x + i * (fig_x + space_x), start_y, fig_x, fig_y])
        plot_ICweight(ax, pattern, thresh, direction='v', ylim=[ymin, ymax], ylabelpad=-1)
        ax.set_xlabel('{}ms'.format(int(df)), fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        ax.tick_params(axis='x', size=2)
        ax.yaxis.get_label().set_fontsize(7)
        ax.set_xticks([])
        if i > 0:
            ax.get_yaxis().set_visible(False)
        if i == 0:
            if not ylabel:
                ax.set_ylabel('')
        ax.set_title('{:.2}'.format(corr[i]), fontsize=6, pad=2)
        ax.spines[['left', 'bottom']].set_visible(False)
        axes.append(ax)

    return axes


def plot_icweight_match_binsize_summary(ax, stim, probe,
                                        datafolder='E:\Congcong\Documents\data\comparison\data-pkl'):
    probe_region = {'H31x64': 'MGB', 'H22x32': 'A1'}
    region = probe_region[probe]
    files = glob.glob(os.path.join(datafolder, '*{}*-{}-ic_match_tbins.pkl'.format(probe, stim)))
    n_ne = np.zeros(7)
    n_match = np.zeros(7)
    for file in files:
        with open(file, 'rb') as f:
            ic_matched = pickle.load(f)

        n_ne = n_ne + np.array(ic_matched['n_ne'])
        for idx_ne in range(ic_matched['n_ne'][0]):
            p = ic_matched['p'][idx_ne]
            idx_sig = np.where(p > .05)[0]
            if not any(idx_sig):
                idx_sig = 7
            else:
                idx_sig = idx_sig[0]
            n_match[1:idx_sig + 1] = n_match[1:idx_sig + 1] + 1
    n_match[0] = n_ne[0]
    df = ic_matched['df']
    df = [x // 2 for x in df]
    ax.bar(range(7), n_ne, color=eval(f'{region}_color[1]'))
    ax.bar(range(7), n_match, color=eval(f'{region}_color[0]'))
    ax.set_xticks(range(7))
    ax.set_xticklabels(df)
    ax.set_xlabel('bin size (ms)')
    ax.set_ylabel('# of cNEs')

    ax.set_title(region, color=eval(f'{region}_color[0]'), weight='bold')


def plot_icweight_corr_vs_binsize_summary(ax, jsonfile, property='corr', stim='spon'):
    """
    violin plot of icweight corr for different binsizes. The 2 halves of violins represent MGB and A1.
    """
    data = pd.read_json(jsonfile)
    data = data[data.stim == stim]
    if property == 'corr':
        data_MGB = data[data.probe == 'H31x64'].groupby('df')['corr'].apply(list)
        data_A1 = data[data.probe == 'H22x32'].groupby('df')['corr'].apply(list)
        thresh = 0.9
    else:
        if property == 'member_overlap_prc':
            data['overlap_prc'] = data['n_member_overlap'] / data['n_member']
        elif property == 'member_overlap_prc_ref':
            data['overlap_prc'] = data['n_member_overlap'] / data['n_member_ref']
        elif property == 'member_overlap_prc_all':
            data['overlap_prc'] = data['n_member_overlap'] / data['n_member_all']
        data_MGB = data[data.probe == 'H31x64'].groupby('df')['overlap_prc'].apply(list)
        data_A1 = data[data.probe == 'H22x32'].groupby('df')['overlap_prc'].apply(list)
        thresh = 0.8
    positions = [0, 1, 3, 4, 5, 6]

    for region in ('MGB', 'A1'):
        print(region)
        data = eval(f'data_{region}')
        for vals in data:
            print(np.mean(np.array(vals) >= thresh))
    v1 = ax.violinplot(data_MGB, points=100, positions=positions, showextrema=False, widths=.8)
    set_violin_half(v1, half='l', color=MGB_color[0])

    v2 = ax.violinplot(data_A1, points=100, positions=positions, showextrema=False, widths=.8)
    set_violin_half(v2, half='r', color=A1_color[0])
    ax.set_ylim([0, 1])
    plt.setp(ax.collections, alpha=1)
    ax.legend([v1['bodies'][0], v2['bodies'][0]], ['MGB', 'A1'],
              fontsize=6, fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xticks(range(7))
    ax.set_xticklabels([2, 5, 10, 20, 40, 80, 160])
    ax.set_xlabel('Bin size (ms)')
    for binsize in [4, 10, 40, 80, 160, 320]:
        _, p = stats.mannwhitneyu(data_MGB[binsize], data_A1[binsize])
        print('binsize', binsize/2, 'ms', p*6)


def set_violin_half(v, half='l', color='r'):
    for b in v['bodies']:
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        if half == 'l':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        elif half == 'r':
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(color)


def plot_member_ccg_binsize(axes, jsonfile, df, stim, probe):
    data = pd.read_json(jsonfile)
    ccg_fieldnames = ('ccg_overlap', 'ccg_extra', 'ccg_ref')
    data_subset = data[(data.stim == stim) & (data.probe == probe) & (data.df == df)]
    ccg_all = [] # both, 160ms only, 10ms only
    for i in range(3):
        ccg = list(eval('data_subset.{}'.format(ccg_fieldnames[i])))
        ccg = np.concatenate([x for x in ccg if x])
        ccg = stats.zscore(ccg, axis=1)
        plot_xcorr_avg(axes[i], ccg)
        ccg_all.append(ccg)
    # both vs 160 only
    plot_xcorr_sig(axes[1], ccg_all[0], ccg_all[1])
    plot_xcorr_sig(axes[1], ccg_all[0], ccg_all[2])


def plot_waveform(ax, waveform_mean, waveform_std, color='k', color_shade='lightgrey'):
    x = range(waveform_mean.shape[0])
    ax.fill_between(x, waveform_mean + waveform_std, waveform_mean - waveform_std, color=color_shade)
    ax.plot(x, waveform_mean, color=color, linewidth=.6)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim([x[0], x[-1]])


def plot_session_waveforms(units):
    nrows = int(np.ceil(np.sqrt(len(units))))
    fig, axes = plt.subplots(nrows, nrows)
    axes = axes.flatten()
    for i in range(len(units)):
        ax = axes[i]
        unit = units[i]
        chan = unit.chan
        idx = np.where(unit.adjacent_chan == chan)[0][0]
        waveform_mean = unit.waveforms_mean[idx]
        waveform_std = unit.waveforms_std[idx]
        plot_waveform(ax, waveform_mean, waveform_std)
        ax.set_title(unit.unit)


def plot_session_ccg(spktrains, maxlag=40):
    n = len(spktrains)
    nrows = int(np.ceil(np.sqrt(n * (n - 1) / 2)))
    fig, axes = plt.subplots(nrows, nrows)
    axes = axes.flatten()
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            ax = axes[c]
            c += 1
            spktrain1 = spktrains[i]
            spktrain2 = np.roll(spktrains[j], -maxlag)
            spktrain2 = spktrain2[:-2 * maxlag]
            xcorr = np.correlate(spktrain1, spktrain2, mode='valid')
            ax.bar(np.arange(-20, 20.1, .5), xcorr, color='k')
            ax.set_xlim([-20, 20])
            ax.set_xlabel('Lag(ms)')
            ax.set_title(f'{i}-{j}')


# +++++++++++++++++++++++++++++++++++++++++++++++ UP/DOWN state ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# raster plot of MU and ne spikes, firing rate of multi-neuron
def plot_raster_fr_prob(session, ne, up_down, figfolder, plot_window=None, stim='spon'):
    probdata = session.probdata
    chan_position = np.array([probdata['xcoords'], probdata['ycoords']])
    chan_order = np.lexsort((chan_position[0, :], chan_position[1, :]))

    for cne_idx in ne.ne_members:
        fig, axes = plt.subplots(3, 1, figsize=(8, 4))

        t_start, t_end = plot_up_ne_spikes(axes[1:], session, ne, up_down, cne_idx, plot_window=None, stim='spon')

        # plot multiunit data
        spiketimes_mu = up_down['spiketimes_mu']
        chan_mu = up_down['chan_mu']
        chan_order_mu = np.array([chan_order[int(x)] for x in chan_mu])
        axes[0].scatter(spiketimes_mu, chan_order_mu, c='k', s=1)
        axes[0].set_xlim([t_start, t_end])
        axes[0].set_ylim([-1, 64])
        axes[0].set_xticklabels([])
        axes[0].set_ylabel('channel #')
        axes[0].set_title('MU')

        plt.tight_layout()
        fig.savefig(os.path.join(figfolder, '{}-{}um-cNE_{}.jpg'.format(session.exp, session.depth, cne_idx)), dpi=300)
        plt.close()


def plot_up_ne_spikes(axes, session, ne, up_down, cne_idx, plot_window=None, stim='spon'):
    # find the 5s with most ne spikes
    if plot_window is None:
        ne_spikes = ne.ne_units[cne_idx].spiketimes
        nspk, edges = np.histogram(ne_spikes, bins=up_down['edges_' + stim][::500])
        idx = np.argmax(nspk)
        t_start = edges[idx]
        t_end = edges[idx + 1]
    else:
        t_start, t_end = plot_window

    # plot MU fring rate and predicted state
    fr_mu = up_down['fr_mu_' + stim]
    fr_edges = np.array(up_down['edges_' + stim])
    fr_centers = (fr_edges[1:] + fr_edges[:-1]) / 2
    axes[0].plot(fr_centers, fr_mu, c='k', linewidth=.5)
    axes[0].set_xlim([t_start, t_end])
    idx_tstart = np.where(fr_centers > t_start)[0][0]
    idx_tend = np.where(fr_centers < t_end)[0][-1]
    ymax = max(fr_mu[idx_tstart:idx_tend])
    axes[0].set_ylim([0, ymax * 1.1])
    axes[0].set_ylabel('MU FR (Hz)')
    axes[0].set_ylim([0, 300])
    axes[0].set_yticks(range(0, 301, 100))
    axes[0].set_yticklabels(range(0, 301, 100))
    axes[0].set_xticks(np.arange(t_start, t_end + 1, 1000))
    axes[0].set_xticklabels(range(11))
    axes[0].set_xlabel('Time (s)')
    # plot member neuron spikes
    members = ne.ne_members[cne_idx]
    units = [session.units[x] for x in members]
    plot_raster(axes[1], units, linewidth=.5)
    plot_raster(axes[1], ne.member_ne_spikes[cne_idx], color='r', linewidth=.5)
    # plot nespike
    axes[1].eventplot(ne.ne_units[cne_idx].spiketimes, lineoffsets=len(members) + 1,
                      linelengths=0.5, colors='r', linewidth=.5)
    axes[1].set_xlim([t_start, t_end])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].spines[['bottom', 'left']].set_visible(False)

    up_interval = up_down['up_interval_' + stim]
    idx_tstart = np.where(up_interval[0] > t_start)[0][0]
    idx_tend = np.where(up_interval[0] < t_end)[0][-1]
    plot_up_interval_shade(axes[1], up_interval[:, idx_tstart - 1:idx_tend + 2], ylim=[0, len(members) + 2])
    return t_start, t_end


def plot_up_interval_shade(ax, up_interval, ylim=[0, 1]):
    for start, end in zip(*up_interval):
        p = mpl.patches.Rectangle((start, ylim[0]),
                                  end - start, ylim[1] - ylim[0], alpha=.5, color='orange')
        ax.add_patch(p)


def plot_p_up_corr_fr(ax, datafolder, figfolder):
    files = glob.glob(datafolder + r'\*-up_down.pkl', recursive=False)

    corr = []
    up_duration = []
    for idx, file in enumerate(files):

        with open(file, 'rb') as f:
            up_down = pickle.load(f)

        if 'p_up' not in up_down:
            continue
        p_up = up_down['p_up']
        fr = up_down['fr_mu']
        up_interval = up_down['up_interval']
        up_dur = up_interval[1] - up_interval[0]
        corr.append(np.corrcoef(p_up, fr)[0][1])
        up_duration.append(up_dur.mean())
    corr = np.array(corr)
    up_duration = np.array(up_duration)


def plot_ne_event_up_percent(ax, stim='spon',
                             datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl\up_down_spon'):
    files = glob.glob(datafolder + r'\*-20dft-{}.pkl'.format(stim), recursive=False)
    p_up = []
    region = []
    for idx, file in enumerate(files):

        with open(file, 'rb') as f:
            ne = pickle.load(f)

        if 'H31x64' in file:
            region_flag = 1
        else:
            region_flag = 0
        for idx, cne in enumerate(ne.ne_units):
            n_event = len(cne.spiketimes)
            n_event_up = len(cne.spiketimes_up)
            p_up.append(n_event_up / n_event * 100)
            region.append(region_flag)

    df = pd.DataFrame({'region': ['MGB' if x == 1 else 'A1' for x in region], 'up_percent': p_up})
    boxplot_scatter(ax, x='region', y='up_percent', data=df, order=['MGB', 'A1'],
                    hue='region', palette=[MGB_color[0], A1_color[0]], hue_order=['MGB', 'A1'],
                    jitter=.3)
    ax.set_title('Percent of events\nin hiAct state', pad=1, fontsize=7)
    ax.set_ylabel('cNE events')
    ax.set_xlabel('')
    _, p = stats.mannwhitneyu(df[df['region'] == 'MGB']['up_percent'], df[df['region'] == 'A1']['up_percent'])
    print('C', p)


def plot_ne_spike_prob_up_vs_all(ax, stim='spon',
                                 datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl\up_down_spon'):
    files = glob.glob(datafolder + r'\*-20dft-{}.pkl'.format(stim), recursive=False)
    p_ne = []
    p_all = []
    region = []
    for idx, file in enumerate(files):

        with open(file, 'rb') as f:
            ne = pickle.load(f)

        session = ne.get_session_data()

        if 'H31x64' in file:
            region_flag = 1
        else:
            region_flag = 0

        for idx, members in ne.ne_members.items():
            for m in range(len(members)):
                member = members[m]
                nspk = len(ne.member_ne_spikes[idx][m].spiketimes)
                nspk_up = len(ne.member_ne_spikes[idx][m].spiketimes_up)
                p_ne.append(nspk_up / nspk * 100)

                nspk = len(session.units[member].spiketimes)
                nspk_up = len(session.units[member].spiketimes_up)
                p_all.append(nspk_up / nspk * 100)
                region.append(region_flag)

    df = pd.DataFrame({'region': ['MGB' if x == 1 else 'A1' for x in region], 'up_ne': p_ne, 'up_all': p_all})
    sns.scatterplot(data=df, x='up_all', y='up_ne', alpha=.5, s=marker_size,
                    hue='region', palette=[MGB_color[0], A1_color[0]], hue_order=['MGB', 'A1'])
    handles, _ = ax.legend_.legendHandles, ax.legend_.texts
    for handle in handles:
        handle._sizes = [marker_size]
    ax.legend([handle for handle in handles], ['MGB', 'A1'], fontsize=6,
              handletextpad=0, labelspacing=.1, borderpad=.3,
              fancybox=False, edgecolor='black', loc='lower right')
    plt.plot([-5, 105], [-5, 105], 'k', linewidth=.8)
    plt.xlim([-5, 105])
    plt.ylim([-5, 105])
    ax.set_ylabel('NE spikes')
    ax.set_xlabel('All spikes')
    _, p = stats.wilcoxon(df[df.region == 'MGB']['up_all'], df[df.region == 'MGB']['up_ne'])
    print('D')
    print('MGB', p)
    _, p = stats.wilcoxon(df[df.region == 'A1']['up_all'], df[df.region == 'A1']['up_ne'])
    print('A1', p)


def plot_cv_vs_sd(ax, example,
                  filepath=r'E:\Congcong\Documents\data\comparison\data-summary\firing_rate_parameters.json'):
    df = pd.read_json(filepath)
    sns.scatterplot(data=df, x='silence_density_dmr', y='spkcount_cv_dmr', hue='region',
                    palette=[MGB_color[1], A1_color[1]], hue_order=['MGB', 'A1'],
                    ax=ax, edgecolor='k', s=marker_size)
    sns.scatterplot(data=df[(df['silence_density_dmr'] < .4) & (df['spkcount_cv_dmr'] < .8)],
                    x='silence_density_dmr', y='spkcount_cv_dmr', hue='region',
                    palette=[MGB_color[0], A1_color[0]], hue_order=['MGB', 'A1'],
                    ax=ax, edgecolor='k', s=marker_size)
    handles, _ = ax.legend_.legendHandles, ax.legend_.texts
    for handle in handles:
        handle._sizes = [marker_size]
        handle.set_edgecolor("k")
        handle.set_linewidth(.5)
    ax.legend([handle for handle in handles], ['', '', 'MGB', 'A1'],
              ncol=2, fontsize=6, columnspacing=-1,
              handletextpad=0, labelspacing=.1, borderpad=.3,
              fancybox=False, edgecolor='black', loc='upper left')
    colors = ['brown', 'green', 'purple']
    i = 0
    for sd, cv in zip(*example):
        ax.scatter(sd, cv, color=colors[i], edgecolor='k', s=marker_size, linewidth=.5)
        i += 1
    ax.set_xlim([0, .9])
    ax.set_ylim([0, 2.5])
    ax.set_yticks(np.arange(0, 3, .5))
    ax.set_xticks(np.arange(0, 1, .2))
    ax.plot([.4, .4], [0, 2], 'k--', linewidth=.8)
    ax.plot([0, .9], [.8, .8], 'k--', linewidth=.8)
    ax.set_xlabel('Silence density')
    ax.set_ylabel('MU Rate c.v.')


def plot_raster_fr_sd_cv(axes, up_down, plot_window, T=10, stim='spon'):
    # plot single unit data
    spiketimes_su = up_down['spiketimes_su']
    unit_su = up_down['unit_su']
    unit_unique = np.unique(unit_su)
    unit_unique = dict(zip(unit_unique, range(1, len(unit_unique) + 1)))
    unit_idx_su = np.array([unit_unique[x] for x in unit_su])
    idx = np.logical_and(spiketimes_su >= plot_window[0], spiketimes_su <= plot_window[1])
    axes[0].scatter(spiketimes_su[idx], unit_idx_su[idx], marker='|', c='k', s=50 / len(unit_unique), linewidth=.5)
    axes[0].set_xlim([plot_window[0], plot_window[1]])
    axes[0].set_ylim([0, len(unit_unique) + 1])
    axes[0].set_xticklabels([])
    axes[0].set_ylabel('')
    axes[0].spines[['bottom', 'left']].set_visible(False)
    axes[0].set_ylim([0, np.max(unit_idx_su[idx]) + 1])
    axes[0].set_yticks([])
    axes[0].set_xticks([])

    # plot MU fring rate
    edges = np.arange(plot_window[0], plot_window[1]+1, 1)
    kernel = np.ones(50) / 50
    spkcount, _ = np.histogram(spiketimes_su, edges)
    fr = np.convolve(spkcount, kernel, mode='same') * 1e3
    centers = (edges[1:] + edges[:-1]) / 2
    axes[1].plot(centers / 1e3, fr, c='k', linewidth=.5)
    axes[1].set_xlim(np.array(plot_window) / 1e3)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('MU FR (Hz)')
    axes[1].set_ylim([0, 250])
    axes[1].set_yticks(range(0, 301, 100))
    axes[1].set_yticklabels(range(0, 301, 100))
    axes[1].set_xticklabels(range(6))


# ----------------------------------------- plot fra gradient --------------------------------------------------
def plot_fra_gradient(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-pkl\fra',
                      figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\fra-stack'):
    files = glob.glob(os.path.join(datafolder, '*thresh.pkl'))
    files = files[4:]
    for file in files:
        print(file)
        data = pd.read_pickle(file)
        for probe in ('H31x64', 'H22x32'):
            data_tmp = data[data.probe == probe]
            if probe == 'H31x64':
                fig, axes = plt.subplots(1, 2)
            else:
                fig, axes = plt.subplots(1, 3)
            plot_fra_stack_cf(axes, data_tmp)
            exp = re.search('\d{6}_\d{6}', file).group(0)
            fig.savefig(os.path.join(figfolder, f'{exp}-{probe}-fra.jpg'))
            plt.close()


def plot_fra_stack_cf(axes, data):
    # freq = data.freq.iloc[0]
    probe = data.probe.iloc[0]
    xlabel = 'Freq. (kHz)'
    ylabel = 'Depth (um)'
    top = int(data.position.iloc[0][-1])
    bottom = int(data.position.iloc[-1][-1])
    n_atten = data.tcmat.iloc[0].shape[0]
    depth = np.concatenate(list(data.position))[1::2]
    tc = np.concatenate(list(data.tcmat.apply(lambda x: x[::-1, :])))
    cf = data.cf
    if probe == 'H31x64':
        yticks = range(63 * n_atten + 4, 0, -10 * n_atten)
        yticklabels = range(bottom, top - 1, -10 * 20)
        im = plot_fra_stack(axes[0], tc, xlabel, ylabel, yticks, yticklabels)
        plot_cf_gradient(axes[1], cf, depth, depth[3::10])

    elif probe == 'H22x32':
        tc2 = tc[:32 * n_atten, :]
        tc1 = tc[32 * n_atten:, :]
        yticks = range(31 * n_atten + 4, -1 * n_atten, -8 * n_atten)
        yticklabels = range(bottom, top - 100, -8 * 25)
        plot_fra_stack(axes[0], tc1, xlabel, ylabel, yticks, yticklabels)
        im = plot_fra_stack(axes[1], tc2, xlabel, ylabel, yticks, yticklabels)
        axes[1].set_ylabel('')
        axes[1].set_yticklabels([])
        depth = depth[:32]
        depth = np.concatenate([[depth[0] - 25], depth])
        plot_cf_gradient(axes[2], cf[:32], depth[1:], depth[::8], color='grey')
        plot_cf_gradient(axes[2], cf[32:], depth[1:], depth[::8], color='k')
    axes[-1].invert_yaxis()
    return im


def plot_fra_stack(ax, tc, xlabel, ylabel, yticks, yticklabels):
    tc = tc / np.max(tc)
    im = ax.imshow(tc, aspect='auto', origin='upper', cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels([.5, 4, 32])
    ax.set_ylabel(ylabel)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    return im


def plot_cf_gradient(ax, cf, y, yticks, color='k'):
    cf = cf.apply(float)
    cf = cf.apply(np.log2)
    ax.scatter(cf, y, c=color, s=2)
    ax.set_ylim([min(y) - 20, max(y) + 20])
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set_xlabel('CF (kHz)')
    ax.set_xticks([-1, 2, 5])
    ax.set_xticklabels([.5, 4, 32])


# --------------------------------------- plot figures ----------------------------------------------------------
def figure1(figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary'):
    figsize = [figure_size[1][0], 8.5 * cm]
    fig = plt.figure(figsize=figsize)

    # example FRA gradient MGB
    x_start = .4
    y_start = .08
    x_space = .03
    x_fig = .1
    y_fig = .42

    file = r'E:\Congcong\Documents\data\comparison\data-pkl\fra' \
           + r'\200709_231103-site1-5300um-30db-fra10-H31x64-fs20000-thresh.pkl'
    data = pd.read_pickle(file)
    data = data[data.probe == 'H31x64']
    axes = []
    axes.append(fig.add_axes([x_start, y_start, x_fig, y_fig]))
    axes.append(fig.add_axes([x_start + x_fig + x_space, y_start, x_fig, y_fig]))
    plot_fra_stack_cf(axes, data)

    file = r'E:\Congcong\Documents\data\comparison\data-pkl\fra' \
           + r'\210218_155422-site1-5026um-35db-fra10-H31x64-fs20000-thresh.pkl'
    data = pd.read_pickle(file)
    data = data[data.probe == 'H31x64']
    axes = []
    x_start2 = x_start + 2 * x_fig + x_space + .08
    axes.append(fig.add_axes([x_start2, y_start, x_fig, y_fig]))
    axes.append(fig.add_axes([x_start2 + x_fig + x_space, y_start, x_fig, y_fig]))
    plot_fra_stack_cf(axes, data)
    axes[0].set_ylabel('')

    # example FRA gradient A1
    y_start = .62
    x_space = .03
    y_fig = .28

    file = r'E:\Congcong\Documents\data\comparison\data-pkl\fra' \
           + r'\200710_001625-site2-5300um-30db-fra10-H31x64-fs20000-thresh.pkl'
    data = pd.read_pickle(file)
    data = data[data.probe == 'H22x32']
    axes = []
    axes.append(fig.add_axes([x_start, y_start, x_fig, y_fig]))
    axes.append(fig.add_axes([x_start + x_fig + x_space, y_start, x_fig, y_fig]))
    axes.append(fig.add_axes([x_start2, y_start, x_fig, y_fig]))
    im = plot_fra_stack_cf(axes, data)
    axes[1].set_xlabel('')
    axes[0].set_title('shank 1', fontsize=6, pad=1)
    axes[1].set_title('shank 2', fontsize=6, pad=1, color='grey')
    axins = inset_axes(
        axes[1],
        width="10%",  # width: 5% of parent_bbox width
        height="40%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(1.05, 0.3, 1, 1),
        bbox_transform=axes[1].transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=axins)
    cb.ax.tick_params(axis='y', direction='in')
    cb.ax.set_yticks([0, 1])
    cb.ax.set_yticklabels([0, 'Max'])
    axins.tick_params(axis='both', which='major', labelsize=6)

    fig.savefig(os.path.join(figfolder, 'fig1_1.jpg'), dpi=1000)
    fig.savefig(os.path.join(figfolder, 'fig1_1.pdf'), dpi=1000)
    plt.close()


def figure1_2(datafolder=r'E:\Congcong\Documents\data\comparison', figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary'):
    figsize = [7.5 * cm, 8.1 * cm]
    fig = plt.figure(figsize=figsize)

    # example STRFs
    units = pd.read_json(os.path.join(datafolder, 'single_units.json'))
    exp = 200709232021
    units = units[(units.exp == exp) & (units.probe == 'H31x64')]
    example_idx = [0, 2, 3, 7, 8, 9, 13, 15]
    units = units.iloc[example_idx]
    x_start = .095
    x_fig = .15
    y_fig = .12
    y_space = .03
    y_start = .42
    axes = []
    for i in range(4):
        axes.append(fig.add_axes([x_start, y_start + i * (y_fig + y_space), x_fig, y_fig]))
    for i in range(4):
        axes.append(fig.add_axes([x_start + .65, y_start + i * (y_fig + y_space), x_fig, y_fig]))
    axes = np.array(axes)
    axes = axes[[3, 2, 1, 0, 7, 6, 5, 4]]
    depth = []
    for i in range(8):
        strf = np.array(units.iloc[i].strf)
        depth.append(units.iloc[i].position[-1])
        vmax = np.max(abs(strf))
        im = plot_strf(axes[i],
                       strf,
                       taxis=np.array(units.iloc[i].strf_taxis),
                       faxis=np.array(units.iloc[i].strf_faxis),
                       smooth=False,
                       vmax=vmax)
        if i == 3:
            axes[i].set_xlabel('Time before spikes (ms)', labelpad=1, fontsize=5.5)
            axes[i].set_ylabel('Frequency (kHz)', labelpad=1, fontsize=5.5)
        else:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        if i == 7:
            # add colorbar
            axins = inset_axes(
                axes[i],
                width="10%",  # width: 5% of parent_bbox width
                height="80%",  # height: 50%
                loc="center left",
                bbox_to_anchor=(1.05, 0, 1, 1),
                bbox_transform=axes[i].transAxes,
                borderpad=0,
            )
            cb = fig.colorbar(im, cax=axins)
            cb.ax.set_yticks([-vmax, 0, vmax])
            cb.ax.set_yticklabels([-1, 0, 1])
            cb.set_label('Normalized\nsound power', rotation=270, fontsize=5, labelpad=10)
            axins.tick_params(axis='both', which='major', labelsize=5)
    axes[0].text(15, 50, '#1', fontsize=6)
    axes[2].text(15, 50, '#2', fontsize=6)
    axes[3].text(15, 50, '#3', fontsize=6)

    # plot waveform 
    x_start += .16
    x_fig = .08
    y_shrink = .05
    y_fig -= y_shrink
    y_space += y_shrink
    y_start = .42 + y_shrink
    axes = []
    for i in range(4):
        axes.append(fig.add_axes([x_start, y_start + i * (y_fig + y_space), x_fig, y_fig]))
    for i in range(4):
        axes.append(fig.add_axes([x_start + .35, y_start + i * (y_fig + y_space), x_fig, y_fig]))
    axes = np.array(axes)
    axes = axes[[3, 2, 1, 0, 7, 6, 5, 4]]
    sessionfile = os.path.join(datafolder, r'200709_232021-site1-5300um-20db-dmr-31min-H31x64-fs20000.pkl')
    with open(sessionfile, 'rb') as f:
        session = pickle.load(f)
    units = session.units
    for i in range(8):
        unit = units[example_idx[i]]
        chan = unit.chan
        idx = np.where(unit.adjacent_chan == chan)[0][0]
        waveform_mean = unit.waveforms_mean[idx]
        waveform_std = unit.waveforms_std[idx]
        plot_waveform(axes[i], waveform_mean, waveform_std)
        axes[i].spines[['bottom', 'left']].set_visible(False)
        axes[i].set_xlim([20, 70])

    # probe
    x_start = .48
    x_fig = .03
    y_fig = .5
    ax = fig.add_axes([x_start, y_start - y_shrink, x_fig, y_fig])
    ax.scatter(np.zeros(64), range(4020, 5281, 20), s=3, linewidth=.3,
               facecolors='none', edgecolors='k')
    ax.scatter(np.zeros(8), depth, s=3, linewidth=.3, color='k')
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.plot([-.3, .3], [4000, 4000], 'k', linewidth=.3)
    ax.plot([-.3, -.3], [4000, 5280], 'k', linewidth=.3)
    ax.plot([.3, .3], [4000, 5280], 'k', linewidth=.3)
    ax.plot([-.3, 0], [5280, 5340], 'k', linewidth=.3)
    ax.plot([0, .3], [5340, 5280], 'k', linewidth=.3)
    ax.set_xticks([])
    ax.set_ylim([4000, 5340])
    ax.set_yticks([4000, 5280])
    ax.set_title('Depth (um)', fontsize=6)
    ax.invert_yaxis()

    # example ccg
    x_start = .12
    y_start = .08
    x_fig = .25
    y_fig = .2
    maxlag = 50

    def plot_ccg(u1, u2, ax):
        
        spktrain1 = session.spktrain_dmr[u1, :]
        spktrain2 = np.roll(session.spktrain_dmr[u2, :], -maxlag)
        spktrain2 = spktrain2[:-2 * maxlag]
        xcorr = np.correlate(spktrain1, spktrain2, mode='valid')
        # xcorr = xcorr / (spktrain1.sum() * spktrain2.sum() / 4) * 1e5
        ax.bar(np.arange(-25, 25.1, .5), xcorr, color='k')
        
        spktrain1 = session.spktrain_spon[u1, :]
        spktrain2 = np.roll(session.spktrain_spon[u2, :], -maxlag)
        spktrain2 = spktrain2[:-2 * maxlag]
        xcorr = np.correlate(spktrain1, spktrain2, mode='valid')
        # xcorr = xcorr / (spktrain1.sum() * spktrain2.sum() / 4) * 1e5
        ax.plot(np.arange(-25, 25.1, .5), xcorr, color='grey')
        
        ax.plot([-maxlag / 2, maxlag / 2],
                np.ones(2) * np.mean((xcorr[0:10] + xcorr[-10:]) / 2),
                ls='--', color='r', linewidth=.6)
        ax.plot([0, 0], [0, 25], ls='-', color='r', linewidth=.6)
        ax.set_xlim([-maxlag / 2, maxlag / 2])
        ax.set_xlabel('Lag (ms)')
        ax.set_ylim([0, 20])


    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_ccg(0, 7, ax)
    plot_ccg_spon(0, 7, ax)
    ax.set_title('1 - 3', pad=0)
    ax.set_ylabel('# of spikes')
    ax = fig.add_axes([.5, y_start, x_fig, y_fig])
    plot_ccg(3, 7, ax)
    plot_ccg_spon(3, 7, ax)
    ax.set_title('2 - 3', pad=0)

    fig.savefig(os.path.join(figfolder, 'fig1_2.jpg'), dpi=1000)
    fig.savefig(os.path.join(figfolder, 'fig1_2.pdf'), dpi=1000)
    plt.close()


def figure2(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-pkl',
            figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary',
            summaryfolder: str = r'E:\Congcong\Documents\data\comparison\data-summary'):
    """
    Figure2: groups of neurons with coordinated activities exist in A1 and MGB
    Plot construction procedures for cNEs and  correlated firing around cNE events
    cNE members show significantly higher cross correlation

    Input:
        datafolder: path to *ne-20dft-dmr.pkl files
        figfolder: path to save figure
    Return:
        None
    """
    mpl.rcParams['axes.labelpad'] = 1

    # use example recording to plot construction procedure
    nefile = os.path.join(datafolder, '210119_222752-site2-4800um-20db-dmr-30min-H31x64-fs20000-ne-20dft-spon.pkl')
    with open(nefile, 'rb') as f:
        ne = pickle.load(f)
    session_file = re.sub('-ne-20dft-spon', '', nefile)
    with open(session_file, 'rb') as f:
        session = pickle.load(f)

    figsize = [figure_size[0][0], 13 * cm]
    fig = plt.figure(figsize=figsize)

    # positions for first 2 plots
    ystart = 0.76
    figy = 0.2
    xstart = 0.05
    xspace = 0.06
    figx = 0.14

    # plot correlation matrix
    ax = fig.add_axes([xstart, ystart, figx, figy])
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
    axins.set_title(' corr.', fontsize=6, pad=5)
    cb.ax.set_yticks([-0.1, 0, 0.1])
    cb.ax.set_yticklabels(['-0.1', '0', '0.1'])
    axins.tick_params(axis='both', which='major', labelsize=6)

    # plot eigen values
    ax = fig.add_axes([xstart + figx + xspace + 0.04, ystart, figx, figy])
    corr_mat = np.corrcoef(ne.spktrain)
    thresh = netools.get_pc_thresh(ne.spktrain)
    plot_eigen_values(ax, corr_mat, thresh)

    # plot ICweights - color coded
    ystart = .48
    ax = fig.add_axes([xstart, ystart, figx, figy])
    patterns = ne.patterns

    patterns[0], patterns[1] = np.array(patterns[1]), np.array(patterns[0])

    members = ne.ne_members
    members[0], members[1] = members[1], members[0]
    im = plot_ICweigh_imshow(ax, ne.patterns, ne.ne_members)
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
    axins.set_title('      IC weight', fontsize=6, pad=5)
    cb.ax.set_yticks(np.arange(-.6, .61, .3))
    cb.ax.set_yticklabels([-.6, -.3, .0, .3, .6])
    axins.tick_params(axis='both', which='major', labelsize=6)

    # stem plots for ICweights
    xstart = xstart + figx + xspace + 0.04
    xspace = 0.002
    figx = 0.05
    n_ne, n_neuron = ne.patterns.shape
    thresh = 1 / np.sqrt(n_neuron)
    c = 0
    for i in range(4):
        ax = fig.add_axes([xstart + figx * i + xspace * i, ystart, figx, figy])
        plot_ICweight(ax, ne.patterns[i], thresh, direction='v', ylim=(-0.3, 0.8))
        if i > 0:
            ax.set_axis_off()

    # second row: activities
    ystart = 0.36
    figy = 0.03
    xstart = 0.05
    figx = 0.42
    centers = (ne.edges[:-1] + ne.edges[1:]) / 2
    activity_idx = 5  # 99.5% as threshold
    # reorder units
    c = 0
    i = 1

    # find the 1s with most ne spikes
    ne_spikes = ne.ne_units[i].spiketimes
    nspk, edges = np.histogram(ne_spikes, bins=ne.edges[::(4000 // ne.df)])
    idx = np.argmax(nspk)
    t_start = edges[idx] + .9e3
    t_end = edges[idx + 1]

    # plot activity
    ax = fig.add_axes([xstart, ystart, figx, figy])
    activity_thresh = ne.activity_thresh[i][activity_idx]
    ylim = [-10, 120]
    plot_activity(ax, centers, ne.ne_activity[i], activity_thresh, [t_start, t_end], ylim)
    ax.set_ylabel('activity (a.u.)', fontsize=6)
    ax.set_title('cNE #1', fontweight='bold')

    # plot raster
    ystart = 0.06
    figy = 0.25
    ax = fig.add_axes([xstart, ystart, figx, figy])
    members = ne.ne_members[i]
    for member in members:
        p = mpl.patches.Rectangle((t_start, member + 0.6),
                                  t_end - t_start, 0.8, color='gainsboro')
        ax.add_patch(p)
        c += 1

    plot_raster(ax, session.units, linewidth=.6)
    ax.eventplot(ne.ne_units[i].spiketimes, lineoffsets=n_neuron + 1, linelengths=0.8, colors='r', linewidth=.6)
    plot_raster(ax, ne.member_ne_spikes[i], offset='unit', color='r', linewidth=.6)
    ax.set_xlim([t_start, t_end])
    ax.spines[['bottom', 'left']].set_visible(False)
    ax.set_xticks([])

    # scale bar
    ax.plot([t_start, t_start + 200], [0.1, 0.1], color='k', linewidth=1)
    ax.text(t_start + 20, -0.8, '0.2 s', fontsize=8)
    ax.set_yticks([n_neuron + 1])
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels(['cNE'], fontsize=6, color='r')
    ax.set_ylim([0, n_neuron + 1.5])

    # xcorr plot
    ystart = .38
    xstart = 0.58
    figx = 0.165
    xspace = 0.04
    figy = 0.12
    yspace = 0.02
    ax = []
    for i in range(4):
        y_extra = 0.04 if i < 2 else 0
        for j in range(2):
            ax.append(fig.add_axes([xstart + j * figx + j * xspace,
                                    ystart + (3 - i) * figy + (3 - i) * yspace + y_extra,
                                    figx, figy]))
    ax = np.reshape(np.array(ax), (4, 2))
    xcorr = pd.read_json(os.path.join(summaryfolder, 'member_nonmember_pair_xcorr_filtered.json'))
    xcorr = xcorr[xcorr['stim'] == 'spon']
    #plot_xcorr(fig, ax, xcorr.groupby(by=['region', 'member']))

    # boxplots
    figx = figx + 0.1
    figy = 0.21
    ystart = .1
    my_order = ['MGB_spon_(w)', 'MGB_spon_(o)', 'A1_spon_(w)', 'A1_spon_(o)']

    # box plot for corr
    print('C')
    ax = fig.add_axes([xstart, ystart, figx, figy])
    corr_mean = xcorr[xcorr['xcorr_sig']].groupby(
        by=['exp', 'region', 'stim', 'member'], as_index=False)['corr'].mean()
    corr_mean['region_stim_member'] = corr_mean[['region', 'stim', 'member']].apply(tuple, axis=1)
    corr_mean['region_stim_member'] = corr_mean['region_stim_member'].apply(lambda x: '_'.join([str(y) for y in x]))
    boxplot_scatter(ax=ax, x='region_stim_member', y='corr', data=corr_mean, order=my_order,
                    hue='region_stim_member', palette=list(MGB_color) + list(A1_color), hue_order=my_order,
                    jitter=0.4, legend=False, alpha=.4)
    ax.set_yticks(np.arange(0, .2, .05))
    ax.set_ylabel('Mean\npairwise correlation')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['MGB\n(m)\n34', 'MGB\n(non-m)\n34', 'A1\n(m)\n17', 'A1\n(non-m)\n17'])
    ax.tick_params(axis='x', which='major')
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([-.02, .15])
    ax.set_xlabel('')
    # significance test:
    # Performing two-way ANOVA
    model = ols(
        'corr ~ C(region) + C(member) + \
            C(region):C(member)', data=corr_mean).fit()
    res = sm.stats.anova_lm(model, typ=2)
    print(res)
    # within and without
    print('within vs outside cNE: wilcoxon signedrank test (corrected)')
    p_all = []
    for i in range(2):
        res = stats.wilcoxon(x=corr_mean[corr_mean['region_stim_member'] == my_order[i * 2]]['corr'],
                             y=corr_mean[corr_mean['region_stim_member'] == my_order[i * 2 + 1]]['corr'],
                             alternative='greater')
        p_all.append(res.pvalue)
    p_corrected = multipletests(p_all, method='b')
    p_corrected = p_corrected[1]
    for i in range(2):
        plot_significance_star(ax, p_corrected[i], [2 * i, 2 * i + 1], 0.12, 0.13)
        print(my_order[i * 2].split('_')[0], p_corrected[i])
    # significance test for z-scored CCG peak value: A1 vs MGB
    print('A1 vs MGB: Mann–Whitney U test (corrected)')
    p_all = []
    for i in range(2):
        res = stats.mannwhitneyu(x=corr_mean[corr_mean['region_stim_member'] == my_order[i]]['corr'],
                                 y=corr_mean[corr_mean['region_stim_member'] == my_order[i + 2]]['corr'])
        p_all.append(res.pvalue)
    p_corrected = multipletests(p_all, method='b')
    p_corrected = p_corrected[1]
    for i in range(2):
        plot_significance_star(ax, p_corrected[i], [i, i + 2], 1.2, 1.3)
        print(my_order[i].split('_')[2], p_corrected[i])

    fig.savefig(os.path.join(figfolder, 'fig2.jpg'), dpi=1000)
    fig.savefig(os.path.join(figfolder, 'fig2.pdf'), dpi=1000)
    plt.close()


def figure2b(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-pkl',
            figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary'):
    # E
    # violin plot for correlation value
    fig = plt.figure(figsize=[3.5 * cm, 3.5 * cm])
    ax = fig.add_axes([.17, .17, .8, .8])
    
    xcorr = pd.read_json(
        r'E:\Congcong\Documents\data\comparison\data-summary\member_nonmember_pair_xcorr_filtered.json')
    xcorr = xcorr[(xcorr['stim'] == 'spon') & (xcorr.region == 'MGB')]

    sns.violinplot(ax=ax, data=xcorr, x='member', y='corr', order=['(w)', '(o)'],
                   legend=False, palette=list(MGB_color))
   
    # significance test for correlation value: within vs outside cNE
    _, p = stats.mannwhitneyu(x=xcorr[xcorr['member'] == '(w)']['corr'],
                             y=xcorr[xcorr['member'] == '(o)']['corr'])
    plot_significance_star(ax, p, [0, 1], 0.4, 0.402)
    print('B')
    print('within vs outside cNE: Mann-Witnney U test')
    print('MGB:',p)

    ax.set_xlabel('')
    ax.set_ylabel('Pairwise correlation', labelpad=0)
    ax.set_ylim([-.1, .5])
    fig.savefig(os.path.join(figfolder, 'fig2b.jpg'), dpi=1000)
    fig.savefig(os.path.join(figfolder, 'fig2b.pdf'), dpi=1000)
    plt.close()


def figure3(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-pkl',
            figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary'):
    mpl.rcParams['lines.markersize'] = 8

    figsize = [figure_size[1][0], 12 * cm]
    fig = plt.figure(figsize=figsize)

    x_start = .07
    y_start = .7
    x_space = .01
    x_fig = .05
    y_fig = .25
    y_space = .1
    # plot example - 1
    file = os.path.join(datafolder,
                        '210120_003822-site3-4800um-20db-dmr-31min-H31x64-fs20000-ne-20dft-spon-ic_match_tbins.pkl')
    with open(file, 'rb') as f:
        ic_matched = pickle.load(f)

    axes = plot_icweight_match_binsize_fig(fig,
                                           patterns=ic_matched['patterns'][3, :, :], corr=ic_matched['pearsonr'][3, :],
                                           dfs=ic_matched['df'], ylabel=True, space_x=x_space,
                                           start_x=x_start, start_y=y_start, fig_y=y_fig, fig_x=x_fig)
    axes[0].yaxis.set_label_coords(-.5, 0.5)

    # plot example - 2
    x_start = .57
    file = os.path.join(datafolder,
                        '201005_213847-site5-5105um-20db-dmr-32min-H31x64-fs20000-ne-20dft-spon-ic_match_tbins.pkl')
    with open(file, 'rb') as f:
        ic_matched = pickle.load(f)
    axes = plot_icweight_match_binsize_fig(fig,
                                           patterns=ic_matched['patterns'][1, :, :], corr=ic_matched['pearsonr'][1, :],
                                           dfs=ic_matched['df'], ylabel=True, space_x=x_space,
                                           start_x=x_start, start_y=y_start, fig_y=y_fig, fig_x=x_fig)
    axes[0].set_ylabel('')
    trans = axes[0].get_xaxis_transform()
    axes[0].plot([-1.5, -1], [0, 0], color="k", transform=trans, clip_on=False, linewidth=.8)
    axes[0].annotate('ICweight', xy=(-1.25, -.05), xycoords=trans,
                     ha="center", va="center", fontsize=6)
    axes[0].annotate('0.5', xy=(-1.25, .05), xycoords=trans,
                     ha="center", va="center", fontsize=6)

    # violin plot of correlation values
    jsonfile = r'/Users/hucongcong/Documents/UCSF/data/summary/icweight_corr_binsize.json'
    # correlation values
    x_start = .07
    y_start = .45
    x_fig = .4
    y_fig = .2
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_icweight_corr_vs_binsize_summary(ax, jsonfile, 'corr')
    # member matching percentage
    x_start = .57
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_icweight_corr_vs_binsize_summary(ax, jsonfile, 'member_overlap_prc')

    # ccg of members
    jsonfile = r'/Users/hucongcong/Documents/UCSF/data/summary/member_pair_ccg_binsize.json'
    x_start = .07
    y_start = .23
    x_fig = .25
    y_fig = .15
    x_space = .05
    axes = []
    for i in range(3):
        axes.append(fig.add_axes([x_start + (x_space + x_fig) * i, y_start, x_fig, y_fig]))
    plot_member_ccg_binsize(axes, jsonfile, 320, 'spon', 'H31x64')
    for i in range(3):
        axes[i].set_xticklabels([])
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')

    y_start = .06
    axes = []
    for i in range(3):
        axes.append(fig.add_axes([x_start + (x_space + x_fig) * i, y_start, x_fig, y_fig]))
    plot_member_ccg_binsize(axes, jsonfile, 320, 'spon', 'H22x32')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    axes[1].set_xlabel('Lag (ms)')

    plt.savefig(os.path.join(figfolder, 'fig3.jpg'), dpi=300)
    #plt.savefig(os.path.join(figfolder, 'fig3.pdf'), dpi=300)
    plt.close()


def figure4(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
            summaryfolder = r'E:\Congcong\Documents\data\comparison\data-summary',
            figfolder=r'E:\Congcong\Documents\data\comparison\figure'):
    """
    Figure4: stability of cNEs on spantaneous and sensory-evoked activity blocks
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
    mpl.rcParams['axes.linewidth'] = .6
    mpl.rcParams['lines.linewidth'] = .8


    # use example recording to plot correlation matrix
    ne_file = os.path.join(datafolder, '200821_015617-site6-5655um-25db-dmr-31min-H31x64-fs20000-ne-20dft-split.pkl')
    with open(ne_file, 'rb') as f:
        ne_split = pickle.load(f)

    fig = plt.figure(figsize=[figure_size[0][0], 9 * cm])
    y_start = 0.07
    x_start = 0.05
    # correlation matrix
    ax = fig.add_axes([x_start, y_start, 0.2, 0.4])
    plot_ne_split_ic_weight_corr(ne_split, ax=ax)
    ybox1 = TextArea("spon1", textprops=dict(color=colors_split[1], size=8, rotation=90, ha='left', va='bottom'))
    ybox2 = TextArea("dmr1", textprops=dict(color=colors_split[2], size=8, rotation=90, ha='left', va='bottom'))
    ybox = VPacker(children=[ybox1, ybox2], align="bottom", pad=0, sep=20)
    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0, frameon=False, bbox_to_anchor=(-0.1, .1),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_ybox)

    xbox1 = TextArea("spon2", textprops=dict(color=colors_split[0], size=8, ha='center', va='center'))
    xbox2 = TextArea("dmr2", textprops=dict(color=colors_split[3], size=8, ha='center', va='center'))
    xbox = HPacker(children=[xbox1, xbox2], align="left", pad=0, sep=20)
    anchored_xbox = AnchoredOffsetbox(loc=8, child=xbox, pad=0., frameon=False, bbox_to_anchor=(0.5, 1.03),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_xbox)

    ax.set_ylabel('cNE #', labelpad=9)
    ax.set_xlabel('cNE #', labelpad=1)
    ax.text(1 - .1, 4 + .25, 'i', color='w', weight='bold', fontsize=8)
    ax.text(2 - .2, 5 + .25, 'ii', color='w', weight='bold', fontsize=8)

    # scatter plot for matching cNE ICweights
    x_start = 0.38
    y_start = .6
    fig_y = 0.25
    fig_x = 0.12
    space_x = 0.015
    axes = []
    title_colors = ('brown', 'blue')
    for i in range(2):
        ax = fig.add_axes([x_start + (fig_x + space_x) * i, y_start, fig_x, fig_y])
        axes.append(ax)
        plot_matching_ic_scatter(ax, ne_split, i + 1)
        ax.set_title('i' * (i + 1), pad=15, color=title_colors[i])
    axes[1].set_yticklabels([])
    axes[0].set_xlabel('spon1 / dmr1 / spon')
    axes[0].set_ylabel('spon2 / dmr2 / dmr')

    # example illustration of significance definition
    y_start = 0.07
    fig_x = 0.25
    fig_y = .25
    ax = fig.add_axes([x_start, y_start, fig_x, fig_y])
    corr_null = ne_split['corr_null']['cross']
    corr_real = ne_split['corr']['cross']
    corr_real = corr_real[1:]
    thresh = ne_split['corr_thresh']['cross']
    handles = plot_ne_split_ic_weight_null_corr_panel(ax, thresh, corr_real, corr_null, color=['brown', 'blue'])
    ax.set_ylim([0, .06])
    plt.legend(handles=handles,
               labels=['shuffled correlation', 'significance threshold (p=0.01)', 'cNE pair i', 'cNE pair ii'],
               loc='upper right',
               fontsize=5, fancybox=False, edgecolor='k',
               handletextpad=1, labelspacing=.1, borderpad=.3, bbox_to_anchor=(1, 1.5))

    # distribution of significant correlations
    df = pd.read_json(os.path.join(summaryfolder, 'split_cNE.json'))
    x_start = .72
    fig_y = 0.23
    space_y = .06
    fig_x = 0.25
    bins = np.linspace(0, 1, 26)
    text_x = .1
    axes_f = []
    sig_prc = []
    for i, stim in enumerate(['cross', 'dmr', 'spon']):
        ax = fig.add_axes([x_start, y_start + i * (fig_y + space_y), fig_x, fig_y])
        axes_f.append(ax)
        ax.set_title(stim, fontsize=8, pad=2)
        ax.set_ylim([0, 0.3])
        ax.set_xlim([0, 1])
        ax.set_xticks(np.arange(0, 1.01, .2))
        if i > 0:
            ax.set_xticklabels([])
        n_ne_sig = np.empty(2)
        n_ne = np.empty(2)
        text_y = 0.25
        ii = 0
        region = 'MGB'
        data = df[(df.stim == stim) & (df.region == region)]
        sns.histplot(data=data, x="corr", bins=bins, color=eval('{}_color[0]'.format(region)),
                     element="step", fill=False, stat='probability', ax=ax, linewidth=.8)
        n_unit = len(data)
        ax.hist(data[data.corr_sig]['corr'], bins=bins, weights=(1/n_unit)*np.ones(len(data[data.corr_sig])),
                color=eval('{}_color[0]'.format(region)))
        ax.scatter(data[data.corr_sig]['corr'].mean(), .2, 
                   s=10, marker='v', color=eval('{}_color[1]'.format(region)))
        ax.scatter(data[data.corr_sig]['corr'].median(), .22, 
                   s=10, marker='v', color=eval('{}_color[0]'.format(region)))
        res = stats.bootstrap([np.array(data['corr_sig']).astype(int)], np.mean, random_state=1)

        corr_sig = df[(df.stim == stim) & (df.region == region)]['corr_sig']
        ratio = corr_sig.mean()
        sig_prc.append(list(res.bootstrap_distribution) + [ratio])
        n_ne_sig[ii] = corr_sig.sum()
        n_ne[ii] = len(corr_sig)
        ax.text(text_x, text_y - 0.03 * ii, '{:.1f}%'.format(ratio * 100),
                color=eval('{}_color[0]'.format(region)), fontsize=6)
        
        if i == 0:
            ax.set_xlabel('|Correlation|')
            ax.set_ylabel('')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

        if i == 1:
            ax.set_ylabel('Proportion')
        if i == 2:
            plt.legend(loc='upper right', labels=['A1', 'MGB'], fontsize=6, fancybox=False, edgecolor='k',
                       handletextpad=1, labelspacing=.1, borderpad=.3, bbox_to_anchor=(1, 1.1))

    # comparison of stability of cNEs in MGB and A1
    for i, j in [(0, 1), (0, 2), (1, 2)]:
         prc_diff = np.array(sig_prc[i]) - np.array(sig_prc[j])
         p = sum(prc_diff > 0) / len(prc_diff) * 2
         p = min(p, 2 - p)
         print(min(1, p*3))

    fig.text(0, .955, 'A', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0, .5, 'B', fontsize=fontsize_panel_label, weight='bold')
    fig.text(.32, .955, 'C', fontsize=fontsize_panel_label, weight='bold')
    fig.text(.32, .5, 'D', fontsize=fontsize_panel_label, weight='bold')
    fig.text(.65, .955, 'E', fontsize=fontsize_panel_label, weight='bold')

    plt.savefig(os.path.join(figfolder, 'fig4.jpg'), dpi=300)
    #plt.savefig(os.path.join(figfolder, 'fig4.pdf'), dpi=300)
    plt.close()


def permutation_test(sample1, sample2, nreps=10000):
    diff = np.mean(sample1) - np.mean(sample2)
    sample_all = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    diff_null = np.zeros(nreps)
    rand_idx = np.array(range(len(sample_all)))
    for i in range(nreps):
        np.random.shuffle(rand_idx)
        diff_null[i] = np.mean(sample_all[rand_idx[:n1]]) - np.mean(sample_all[rand_idx[n1:]])
    p = 2 * np.mean(diff_null >= diff)
    return p if p < 1 else 2 - p


def figure8(datafolder=r'E:\Congcong\Documents\data\comparison\data-summary', figfolder=r'E:\Congcong\Documents\data\comparison\figure\summary'):
    figsize = [figure_size[2][0], 11 * cm]
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # panel A: number of cNEs identified on real data vs shuffled data
    fig.text(0, 0.96, 'A', fontsize=fontsize_panel_label, weight='bold')
    ax = axes[0]
    pos = ax.get_position()
    pos.y0 = pos.y0 + .11
    pos.y1 = pos.y1 + .09
    pos.x0 = pos.x0 + .02
    pos.x1 = pos.x1 + .079
    ax.set_position(pos)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plot_num_cne_vs_shuffle(ax, datafolder)

    # panel B: |corr.| distribution of real data and shuffled data
    fig.text(0, 0.6, 'B', fontsize=fontsize_panel_label, weight='bold')
    ne_real = pd.read_json(os.path.join(datafolder, 'cNE_matched.json'))
    ne_shuffled = pd.read_json(os.path.join(datafolder, 'cNE_matched_shuffled.json'))
    bins = np.arange(0, 1.01, 0.05)
    region = ['MGB' if x == 'H31x64' else 'A1' for x in ne_real['probe']]
    ne_real['region'] = region
    region = ['MGB' if x == 'H31x64' else 'A1' for x in ne_shuffled['probe']]
    ne_shuffled['region'] = region

    for i, region in enumerate(('MGB', 'A1')):
        ax = axes[i + 1]
        plot_ne_sig_corr_hist(ax, ne_real, ne_shuffled, region, bins)

        handles, labels = ax.legend_.legendHandles, ax.legend_.texts
        order = [0, 2, 1, 3]
        ax.legend([handles[idx] for idx in order], [labels[idx].get_text() for idx in order],
                  ncol=2, fontsize=6, columnspacing=0.8,
                  frameon=False, loc='upper center', bbox_to_anchor=(.1 - i * .2, 0, 1, 1.1))

        pos = ax.get_position()
        pos.y0 = pos.y0 - .01
        pos.y1 = pos.y1 - .01
        pos.x0 = pos.x0 + .02
        pos.x1 = pos.x1 + .079
        ax.set_position(pos)
        ax.set_xlabel('')
        if i == 0:
            ax.set_xticklabels([])
        ax.get_yaxis().set_label_coords(-0.1, 0.5)

    ax.set_xlabel('|Correlation|')
    plt.savefig(os.path.join(figfolder, 'fig5.jpg'), dpi=300)
    #plt.savefig(os.path.join(figfolder, 'fig5.pdf'), dpi=300)
    plt.close()


def figure5(datafolder=r'E:\Congcong\Documents\data\comparison\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\comparison\figure\summary'):
    figsize = [figure_size[0][0], 10 * cm]
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    stim = 'spon'
    # scatter plot of number of cNE vs number of neurons
    print('A')
    num_ne_vs_num_neuron(axes[0][0], datafolder, stim=stim)
    axes[0][0].set_ylim([0, 8])
    axes[0][0].set_xlim([5, 30])
    # scatter plot of cNE size vs number of neurons
    print('B')
    ne_size_vs_num_neuron(axes[0][1], datafolder, stim=stim, plot_type='raw')
    axes[0][1].set_xlim([5, 30])
    axes[0][1].set_ylim([0, 10])

    # box plot of cNE size vs number of neurons with recordings of certain sizes
    print('C')
    ne_size_vs_num_neuron(axes[0][2], datafolder, stim=stim, plot_type='raw', relative=True, n_neuron_filter=(13, 29))
    # bar plot of number of cNEs each neuron participated in
    print('D')
    num_ne_participate(axes[0][3], datafolder, stim=stim, n_neuron_filter=(13, 29))

    # spacial and frequency distribution
    probe_region = {'H31x64': 'MGB', 'H22x32': 'A1'}
    for i, probe in enumerate(('H31x64', 'H22x32')):
        region = probe_region[probe]
        print(region)
        # pairwise distance
        print('Ei')
        ne_member_distance(axes[1][i], datafolder, stim=stim, probe=probe, direction='vertical', linewidth=1)
        axes[1][i].annotate(region, xy=(.5, 1), xycoords='axes fraction', ha="center", va="center",
                            fontsize=10, color=eval(f'{region}_color[0]'))
        # cNE span
        print('Eii')
        ne_member_span(axes[1][i + 2], datafolder, stim=stim, probe=probe, linewidth=1)
        axes[1][i + 2].annotate(region, xy=(.5, 1.05), xycoords='axes fraction', ha="center", va="center",
                                fontsize=10, color=eval(f'{region}_color[0]'))
        axes[1][i + 2].set_ylim([0, .2])
        axes[1][i + 2].set_yticks(np.arange(0, .21, .05))
        # pairwise frequency difference
        print('Fi')
        ne_member_freq_distance(axes[2][i], datafolder, stim=stim, probe=probe, linewidth=1)
        # cNE freq span
        print('Fii')
        ne_member_freq_span(axes[2][i + 2], datafolder, stim=stim, probe=probe, linewidth=1)
        if i == 1:
            axes[1][i].set_xlabel('')
            axes[1][i + 2].set_xlabel('')
            axes[2][i].set_xlabel('')
            axes[2][i + 2].set_xlabel('')
        else:
            axes[1][i].xaxis.set_label_coords(1.25, -.22)
            axes[1][i + 2].xaxis.set_label_coords(1.25, -.22)
            axes[2][i].xaxis.set_label_coords(1.25, -.22)
            axes[2][i + 2].xaxis.set_label_coords(1.25, -.22)

    for row in axes:
        for ax in row:
            ax.patch.set_alpha(0)

    # adjust layout
    y_adjust = [[.08, .0, -.035], [.08, -.02, -.06]]
    x_adjust = [[-.062, -.02, .03, .08], [-.062, -.02, .03, .08]]
    for i in range(4):
        for j in range(3):
            # panel label for the first row

            ax = axes[j][i]
            pos = ax.get_position()
            pos.y0 = pos.y0 + y_adjust[0][j]
            pos.y1 = pos.y1 + y_adjust[1][j]
            pos.x0 = pos.x0 + x_adjust[0][i]
            pos.x1 = pos.x1 + x_adjust[1][i]
            ax.set_position(pos)

            if i == 0:
                ax.get_yaxis().set_label_coords(-0.24, 0.5)
            elif j > 0:
                ax.set_ylabel('')

            if j == 0:
                ax.get_xaxis().set_label_coords(0.5, -.2)
            elif j == 1 and i < 2:
                ax.set_ylim([0, 0.15])
    # panel label
    y_coord = 0.962
    fig.text(0, y_coord, 'A', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.24, y_coord, 'B', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.49, y_coord, 'C', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.75, y_coord, 'D', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0, 0.6, 'E', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.49, 0.6, 'F', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0, 0.29, 'G', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.49, 0.29, 'H', fontsize=fontsize_panel_label, weight='bold')

    plt.savefig(os.path.join(figfolder, 'fig6.jpg'), dpi=300)
    plt.savefig(os.path.join(figfolder, 'fig6.pdf'), dpi=300)
    plt.close()


def figure6(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-summary',
            figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary'):
    fig = plt.figure(figsize=[figure_size[1][0], 6 * cm])
    y_start = .115
    x_start1 = .065
    x_fig = .14
    x_space = .02
    y_fig = .28
    y_space = .16

    # plot example strf
    axes = []
    for i in range(2):
        for j in range(2):
            axes.append(fig.add_axes([x_start1 + j * (x_fig + x_space),
                                      y_start + i * (y_fig + y_space),
                                      x_fig, y_fig]))
    axes = np.reshape(axes, [2, 2])

    def get_example_data(data, example):
        example_data = data[(data.exp == example[0]) & (data.region == 'MGB') &
                            (data.cNE == example[1]) & (data.member == example[2])]
        return example_data

    file = os.path.join(datafolder, 'subsample_ne_neuron_example.pkl')
    try:
        with open(file, 'rb') as f:
            example = pickle.load(f)
    except:
        data = pd.read_json(os.path.join(datafolder, 'subsample_ne_neuron.json'))
        example = []
        example1 = [191126211411, 0, 4]
        example2 = [200710002633, 2, 9]
        example.append(get_example_data(data, example1))
        example.append(get_example_data(data, example2))
        with open(file, 'wb') as f:
            pickle.dump(example, f)
    taxis, faxis = get_taxis_faxis(datafolder)
    for i in range(2):
        unit = example[i]
        strf = []
        ptd = []
        mi = []
        for spk in ('neuron', 'ne_spike'):
            strf.append(np.array(eval(f'unit.strf_{spk}.values[0]'))[:, :, 0])
            ptd.append(eval(f'unit.ptd_{spk}.values[0][0]'))
            mi.append(eval(f'unit.mi_{spk}.values[0][0]'))
        m = np.max(np.abs(strf))
        for j in range(2):
            strf[j] = strf[j] / m
            im = plot_strf(axes[i][j], strf[j], taxis, faxis, vmax=1)
            if i == 1 and j == 1:
                axins = inset_axes(
                    axes[i][j],
                    width="8%",  # width: 5% of parent_bbox width
                    height="80%",  # height: 50%
                    loc="center left",
                    bbox_to_anchor=(1.05, 0., 1, 1),
                    bbox_transform=axes[i][j].transAxes,
                    borderpad=0,
                )
                cb = plt.colorbar(im, cax=axins)
                cb.ax.tick_params(axis='y', direction='in')
                cb.ax.set_ylabel('Normalized firing rate', fontsize=6, rotation=270, labelpad=6)
                cb.ax.set_yticks([-1, -.5, 0, .5, 1])
                axins.tick_params(axis='both', which='major', labelsize=6)
            if i == 0 and j == 0:
                axes[i][j].set_xlabel('Time before spike (ms)')
                axes[i][j].set_ylabel('Frequency (kHz)')
                axes[i][j].xaxis.set_label_coords(1.25, -.22)
                axes[i][j].yaxis.set_label_coords(-.3, 1.25)
            else:
                axes[i][j].set_xlabel('')
                axes[i][j].set_ylabel('')
                if i > 0:
                    axes[i][j].set_xticklabels([])
                if j > 0:
                    axes[i][j].set_yticklabels([])
            axes[i][j].spines[['top', 'right']].set_visible(True)
            trans = axes[i][j].get_xaxis_transform()
            axes[i][j].annotate('neuron#{} in cNE{}'.format(unit.member.item(), unit.cNE.item()+1), xy=(.1, 1.25), xycoords=trans,
                                ha="left", va="center", fontsize=6)
            axes[i][j].annotate('PTD = {:.2f}'.format(ptd[j]), xy=(.1, 1.15), xycoords=trans,
                                ha="left", va="center", fontsize=6)
            axes[i][j].annotate('MI = {:.2f}'.format(mi[j]), xy=(.1, 1.05), xycoords=trans,
                                ha="left", va="center", fontsize=6)

    axes[1][0].set_title('all spikes', pad=15)
    axes[1][1].set_title('ne spikes', pad=15)

    # plot population result
    y_space = .17
    data = pd.read_json(os.path.join(datafolder, 'subsample_ne_neuron_summary.json'))
    data = data[data.strf_sig > 0]
    data = data[(data.n_events > 100)]
    data['strf_ri_neuron'] = data['strf_ri_neuron_mean']
    data['strf_ri_ne_spike'] = data['strf_ri_ne_spike_mean']

    axes = []
    x_start2 = .55
    x_space = [.02, .07]
    x_fig = [x_fig, x_fig, x_fig / 2]

    for i in range(2):
        for j in range(3):
            axes.append(fig.add_axes([x_start2 + np.sum(x_fig[:j]) + np.sum(x_space[:j]),
                                      y_start + i * (y_fig + y_space),
                                      x_fig[j], y_fig]))
    axes = np.reshape(axes, [2, 3])
    for j, param in enumerate(('mi', 'ptd')):
        gain = {}
        for i, region in enumerate(('MGB', 'A1')):
            data_tmp = data[data.region == region]
            axes[j][i].scatter(data_tmp[f'{param}_neuron'], data_tmp[f'{param}_ne_spike'],
                               s=2, alpha=.6, facecolor=eval(f'{region}_color[0]'), edgecolor="none")
            _, p = stats.wilcoxon(x=data_tmp[f'{param}_neuron'], y=data_tmp[f'{param}_ne_spike'])
            n = len(data_tmp[f'{param}_neuron'])
            if param == 'mi':
                lim = [0, 3]
                axes[j][i].set_xticks(range(4))
                axes[j][i].set_yticks(range(4))
                xlabel = 'Neuron STRF MI (bits/spike)'
                ylabel = 'ne spike STRF MI\n(bits/spike)'
                add_p_val(axes[j][i], .25, 2.5, p)
            elif param == 'ptd':
                axes[j][i].set_xticks(range(0, 26, 5))
                axes[j][i].set_yticks(range(0, 26, 5))
                lim = [0, 25]
                xlabel = 'Neuron STRF PTD'
                ylabel = 'ne spike STRF PTD'
                add_p_val(axes[j][i], 2, 22, p)
                axes[j][i].text(2, 18, f'n = {n}', color='k', fontsize=6)
                axes[j][i].set_title(region, color=eval(f'{region}_color[0]'))
            else:
                lim = [0, 3]
            axes[j][i].set_xlim(lim)
            axes[j][i].set_ylim(lim)
            axes[j][i].plot(lim, lim, 'k')
            if i == 0:
                axes[j][i].set_xlabel(xlabel)
                axes[j][i].set_ylabel(ylabel)
            else:
                axes[j][i].set_yticklabels([])
            axes[j][i].xaxis.set_label_coords(1.25, -.22)

        # gain
        data['gain'] = data[f'{param}_ne_spike'] - data[f'{param}_neuron']
        _, p = stats.mannwhitneyu(data[data.region == 'MGB']['gain'], data[data.region == 'A1']['gain'])
        print('gain: p=', p)
        boxplot_scatter(axes[j][2], x='region', y='gain', data=data, order=('MGB', 'A1'),
                        hue='region', palette=[MGB_color[0], A1_color[0]], hue_order=('MGB', 'A1'), scatter=False)
        if j == 1:
            plot_significance_star(axes[j][2], p, [1, 2], 30, 31, linewidth=.8, fontsize=10)
        else:
            plot_significance_star(axes[j][2], p, [1, 2], 3.5, 3.6, linewidth=.8, fontsize=10)
        axes[j][2].set_xlabel('')

    axes[0][2].set_ylabel('NE spike\nSTRF MI gain', fontsize=6.5)
    axes[0][2].set_ylim([-.5, 1])
    axes[1][2].set_ylabel('NE spike\nSTRF PTD gain', fontsize=6.5)
    axes[1][2].set_ylim([-10, 10])

    y = .945
    fig.text(0, y, 'A', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.5, y, 'B', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.5, .47, 'C', fontsize=fontsize_panel_label, weight='bold')

    plt.savefig(os.path.join(figfolder, 'fig6.jpg'), dpi=1000)
    plt.savefig(os.path.join(figfolder, 'fig6.pdf'), dpi=1000)

    plt.close()


def add_p_val(ax, x, y, p, c='k'):
    if p < 1e-3:
        ax.text(x, y, 'p = {:.1e}'.format(p), color=c, fontsize=6)
    else:
        ax.text(x, y, 'p = {:.3f}'.format(p), color=c, fontsize=6)



def figure7(datafolder: str = r'E:\Congcong\Documents\data\comparison\data-summary',
            figfolder: str = r'E:\Congcong\Documents\data\comparison\figure\summary'):
    
    fig = plt.figure(figsize=[11.6 * cm, 10 * cm])
    # plot population result
    data = pd.read_json(os.path.join(datafolder, 'subsample_ne_neuron_summary.json'))
    data = data[data.strf_sig > 0]
    data = data[(data.n_events > 100)]
    data = data[(data.mi_neuron > 0) & (data.mi_ne_spike > 0)]
    data['strf_ri_neuron'] = data['strf_ri_neuron_mean']
    data['strf_ri_ne_spike'] = data['strf_ri_ne_spike_mean']

    y_start = [.5, .75]
    fig_y = .2
    for j, param in enumerate(('mi', 'ptd')):
        gain = {}
        region = 'A1'
        data_tmp = data[data.region == region]
        y = y_start[j]
        ax = fig.add_axes([.05, y, .2, fig_y])
        ax.scatter(data_tmp[f'{param}_neuron'], data_tmp[f'{param}_ne_spike'],
                           s=2, alpha=.6, facecolor=eval(f'{region}_color[0]'), edgecolor="none")
        _, p = stats.wilcoxon(x=data_tmp[f'{param}_neuron'], y=data_tmp[f'{param}_ne_spike'])
        diff = data_tmp[f'{param}_ne_spike'] - data_tmp[f'{param}_neuron']
        n = len(data_tmp[f'{param}_neuron'])
        if param == 'mi':
            lim = [0, 3]
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            xlabel = 'Neuron STRF MI (bits/spike)'
            ylabel = 'ne spike STRF MI\n(bits/spike)'
            add_p_val(ax, .25, 2.5, p)
        elif param == 'ptd':
            ax.set_xticks(range(0, 26, 5))
            ax.set_yticks(range(0, 26, 5))
            lim = [0, 25]
            xlabel = 'Neuron STRF PTD'
            ylabel = 'ne spike STRF PTD'
            add_p_val(ax, 2, 22, p)
            ax.text(2, 18, f'n = {n}', color='k', fontsize=6)
            ax.set_title(region, color=eval(f'{region}_color[0]'))
        else:
            lim = [0, 3]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.plot(lim, lim, 'k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        gain = (data_tmp[f'{param}_ne_spike'] - data_tmp[f'{param}_neuron'])
        print(region, param, 'gain, median=', np.median(gain[region]),
              'mean=', np.mean(gain[region]), 'std=', np.std(gain[region]))

        # single unit properties
        plot_data = [list(data[data.region == region][f'{param}_neuron']) for region in ('MGB', 'A1')]
        v = axes[j][2].violinplot(plot_data,
                                  points=100, showextrema=False, widths=.8)
        colors = [MGB_color[0], A1_color[0]]
        for i, pc in enumerate(v['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(1)
            pc.set_edgecolor(colors[i])
        _, p = stats.mannwhitneyu(plot_data[0], plot_data[1])
        if j == 1:
            plot_significance_star(axes[j][2], p, [1, 2], 30, 31, linewidth=.8, fontsize=10)
        else:
            plot_significance_star(axes[j][2], p, [1, 2], 3.5, 3.6, linewidth=.8, fontsize=10)
        print('su: p=', p)

        # gain
        gain = [list(gain[key]) for key in ('MGB', 'A1')]
        _, p = stats.mannwhitneyu(gain[0], gain[1])
        print('gain: p=', p)
        v = axes[j][3].violinplot(gain, points=100, showextrema=False, widths=.8)
        for i, pc in enumerate(v['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(1)
            pc.set_edgecolor(colors[i])
        if j == 1:
            plot_significance_star(axes[j][3], p, [1, 2], 30, 31, linewidth=.8, fontsize=10)
        else:
            plot_significance_star(axes[j][3], p, [1, 2], 3.5, 3.6, linewidth=.8, fontsize=10)
        axes[j][2].set_xlabel('')

    axes[0][2].set_ylabel('Neuron STRF MI')
    axes[0][2].set_ylim([0, 4])
    axes[1][2].set_ylabel('Neuron STRF PTD')
    axes[1][2].set_ylim([0, 35])

    axes[0][3].set_ylabel('NE spike\nSTRF MI gain')
    axes[0][3].set_ylim([-1, 2])
    axes[1][3].set_ylabel('NE spike\nSTRF PTD gain')
    axes[1][3].set_ylim([-20, 20])

     print('E')
    for region in ('MGB', 'A1'):
        sig1 = df[(df.stim == 'cross') & (df.region == region)]['corr_sig']
        for stim in ('dmr', 'spon'):
            sig2 = df[(df.stim == stim) & (df.region == region)]['corr_sig']
            p = permutation_test(sig1, sig2)
            print(region, stim, p)

    # test for difference in IC weights correlation values
    def corr_comparison(df):
        model = ols('corr ~ C(region) + C(stim) + C(region):C(stim)', data=df).fit()
        anova_df = sm.stats.anova_lm(model)
        print(anova_df)
        print('MGB vs A1')
        for stim in ('spon', 'dmr', 'cross'):
            _, p = stats.mannwhitneyu(df[(df.stim == stim) & (df.region == 'MGB')]['corr'],
                                      df[(df.stim == stim) & (df.region == 'A1')]['corr'])
            print(stim, p * 7)
        for region in ('MGB', 'A1'):
            for stim in ('dmr', 'spon'):
                new_df = []
                for dmr_first in (True, False):
                    data_cross = df[(df.region == region) & (df.stim == 'cross') & (df.dmr_first == dmr_first)]
                    data_stim = df[(df.region == region) & (df.stim == stim) & (df.dmr_first == dmr_first)]
                    if (dmr_first and stim == 'dmr') or (not dmr_first and stim == 'spon'):
                        new_df.append(pd.merge(data_cross, data_stim[['exp', 'idx1', 'idx2', 'corr']],  how='left',
                                        left_on=['exp', 'idx1'], right_on=['exp', 'idx2']))
                    elif (dmr_first and stim == 'spon') or (not dmr_first and stim == 'dmr'):
                        new_df.append(pd.merge(data_cross, data_stim[['exp', 'idx1', 'idx2', 'corr']], how='left',
                                       left_on=['exp', 'idx2'], right_on=['exp', 'idx1']))
                new_df = pd.concat(new_df)
                _, p = stats.wilcoxon(new_df['corr_x'], new_df['corr_y'], nan_policy='omit')
                print(region, stim, 'vs cross', p*7)

    print('All')
    corr_comparison(df)
    print('Sig')
    corr_comparison(df[df.corr_sig])


    plt.savefig(os.path.join(figfolder, 'fig7.jpg'), dpi=1000)
    plt.savefig(os.path.join(figfolder, 'fig7.pdf'), dpi=1000)

    plt.close()


def figure9():
    data_folder = r'E:\Congcong\Documents\data\comparison\data-pkl\up_down_spon'
    up_down_folder = r'E:\Congcong\Documents\data\comparison\data-pkl\up_down'
    figsize = [figure_size[1][0], 10 * cm]
    fig = plt.figure(figsize=figsize)
    x_start = .05
    y_start = .1
    x_fig = .19
    x_space = .03
    y_fig = .35
    y_space = .15

    sd_example = []
    cv_example = []
    # example raster
    # no up/down
    example_file = '191126_211411-site5-5214um-20db-dmr-16min-H31x64-fs20000-up_down.pkl'
    with open(os.path.join(up_down_folder, example_file), 'rb') as f:
        up_down = pickle.load(f)
    plot_window = np.array([14e3, 20e3])
    axes = []
    axes.append(fig.add_axes([x_start, y_start + .5 * y_fig, x_fig, y_fig / 2]))
    axes.append(fig.add_axes([x_start, y_start, x_fig, 0.4 * y_fig]))
    plot_raster_fr_sd_cv(axes, up_down, plot_window)
    sd_example.append(up_down['silence_density_dmr'].mean())
    cv_example.append(up_down['spkcount_cv_dmr'].mean())

    # weak up/down
    example_file = '201005_213847-site5-5105um-20db-dmr-32min-H31x64-fs20000-up_down.pkl'
    with open(os.path.join(up_down_folder, example_file), 'rb') as f:
        up_down = pickle.load(f)
    plot_window = np.array([978e3, 983e3])
    axes = []
    # raster [0]
    axes.append(fig.add_axes([x_start + x_fig + .5 * x_space, y_start + .5 * y_fig, x_fig, y_fig / 2]))
    axes.append(fig.add_axes([x_start + x_fig + .5 * x_space, y_start, x_fig, 0.4 * y_fig]))
    plot_raster_fr_sd_cv(axes, up_down, plot_window)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([])
    sd_example.append(up_down['silence_density_dmr'].mean())
    cv_example.append(up_down['spkcount_cv_dmr'].mean())

    # strong up/down
    example_file = '210603_201616-site2-4800um-20db-dmr-32min-H31x64-fs20000-up_down.pkl'
    with open(os.path.join(up_down_folder, example_file), 'rb') as f:
        up_down = pickle.load(f)
    plot_window = np.array([17e3, 22e3])
    axes = []
    axes.append(fig.add_axes([x_start + x_fig + .5 * x_space, y_start + 1.5 * y_fig + y_space, x_fig, y_fig / 2]))
    axes.append(fig.add_axes([x_start + x_fig + .5 * x_space, y_start + y_fig + y_space, x_fig, 0.4 * y_fig]))
    plot_raster_fr_sd_cv(axes, up_down, plot_window)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([])
    sd_example.append(up_down['silence_density_dmr'].mean())
    cv_example.append(up_down['spkcount_cv_dmr'].mean())

    # scatter plot of cv vs sd
    ax = fig.add_axes([x_start, y_start + y_fig + y_space, x_fig, y_fig])
    summary_file = os.path.join(datafolder, 'firing_rate_parameters.json')
    plot_cv_vs_sd(ax, example=[sd_example, cv_example], filepath=summary_file)

    # stim of recordings without strong up/down oscilation
    print('B')
    x_start = .5
    fr_summary = pd.read_json(summary_file)
    included = fr_summary[(fr_summary.silence_density_dmr < .4) & (fr_summary.spkcount_cv_dmr < .8)]
    # number of cNEs detected
    num_ne_df = pd.read_json(os.path.join(datafolder, 'num_ne_data_vs_shuffle_fig8.json'))
    num_ne_df = num_ne_df[num_ne_df.stim == 'dmr']
    mi_data = pd.read_json(os.path.join(datafolder, 'subsample_ne_neuron_summary.json'))
    mi_data = mi_data[mi_data.strf_sig > 0]
    mi_data = mi_data[(mi_data.n_events > 100)]
    for i, region in enumerate(('A1', 'MGB')):
        ax = fig.add_axes([x_start, y_start + i * (y_fig + y_space), x_fig, y_fig])
        mi_tmp = mi_data[mi_data.region == region]
        exp = included[included.region == region]['exp']
        data_tmp = mi_tmp.loc[mi_tmp['exp'].isin(exp)]
        # number of cNEs detected
        num_ne_tmp = num_ne_df[num_ne_df.region == region]
        num_ne_tmp = num_ne_tmp.loc[num_ne_tmp['exp'].isin(exp)]

        ax.scatter(data_tmp[f'mi_neuron'], data_tmp[f'mi_ne_spike'],
                   s=5, alpha=.6, facecolor=eval(f'{region}_color[0]'), edgecolor="none")
        ax.plot([0, 3], [0, 3], 'k')
        _, p = stats.wilcoxon(x=data_tmp[f'mi_neuron'], y=data_tmp[f'mi_ne_spike'])
        n = len(data_tmp[f'mi_neuron'])
        print(f'{region}(n={n}): p={p}')
        ax.set_xlim([0, 3])
        ax.set_ylim([0, 3])
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_title(region, color=eval(f'{region}_color[0]'))
        ax.text(.25, 2, f'n = {n}', color='k', fontsize=6)
        add_p_val(ax, .25, 2.5, p)
        if i == 0:
            ax.set_xlabel('Neuron STRF MI (bits/spike)')
            ax.set_ylabel('NE spike STRF MI (bits/spike)')

    # stability across stimulus conditions
    print('C')
    x_start = .75
    fig_y = 0.23
    space_y = .06
    fig_x = 0.22
    included = fr_summary[(fr_summary.silence_density_dmr < .4) & (fr_summary.spkcount_cv_dmr < .8)
                            &(fr_summary.silence_density_spon < .4) & (fr_summary.spkcount_cv_spon < .8)]
    df = pd.read_json(os.path.join(datafolder, 'split_cNE.json'))
    bins = np.linspace(0, 1, 26)
    text_x = .1
    axes_f = []
    samples = {}
    for i, stim in enumerate(['dmr', 'spon', 'cross']):
        ax = fig.add_axes([x_start, y_start + i * (fig_y + space_y), fig_x, fig_y])
        axes_f.append(ax)
        ax.set_title(stim, fontsize=8, pad=2)
        ax.set_ylim([0, 0.3])
        ax.set_xlim([0, 1])
        ax.set_xticks(np.arange(0, 1.01, .2))
        if i > 0:
            ax.set_xticklabels([])
        n_ne_sig = np.empty(2)
        n_ne = np.empty(2)
        text_y = 0.25
        for ii, region in enumerate(['MGB', 'A1']):
            data = df[(df.stim == stim) & (df.region == region)]
            exp = included[included.region == region]['exp']
            data = data.loc[data['exp'].isin(exp)]
            samples[f'{stim}_{region}'] = data['corr_sig']
            sns.histplot(data=data, x="corr", bins=bins, color=eval('{}_color[0]'.format(region)),
                         element="step", fill=False, stat='probability', ax=ax, linewidth=.8)
            n_unit = len(data)
            sns.histplot(data=data[data.corr_sig], x="corr", bins=bins, color=eval('{}_color[0]'.format(region)),
                         ec=eval('{}_color[0]'.format(region)), fill=True, weights=1 / n_unit, ax=ax, linewidth=.8)
            res = stats.bootstrap([np.array(data['corr_sig']).astype(int)], np.mean, random_state=1)

            corr_sig = data['corr_sig']
            ratio = corr_sig.mean()
            n_ne_sig[ii] = corr_sig.sum()
            n_ne[ii] = len(corr_sig)
            ax.text(text_x, text_y - 0.03 * ii, '{:.1f}% ({}/{})'.format(ratio * 100, corr_sig.sum(), len(corr_sig)),
                    color=eval('{}_color[0]'.format(region)), fontsize=6)
        p = permutation_test(samples[f'{stim}_MGB'], samples[f'{stim}_A1'])
        print(stim, 'MGB vs A1', p)
        if i == 0:
            ax.set_xlabel('|Correlation|')
            ax.set_ylabel('')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

        if i == 1:
            ax.set_ylabel('Proportion')
        if i == 2:
            plt.legend(loc='upper right', labels=['MGB', 'A1'], fontsize=6, fancybox=False, edgecolor='k',
                       handletextpad=1, labelspacing=.1, borderpad=.3, bbox_to_anchor=(1, 1.1))

    for region in ('MGB', 'A1'):
        for stim in ('spon', 'dmr'):
            p = permutation_test(samples[f'{stim}_{region}'], samples[f'cross_{region}'])
            print(region, f'{stim} vs cross', p)

    fig.text(0, .95, 'A', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.45, .95, 'B', fontsize=fontsize_panel_label, weight='bold')
    fig.text(0.7, .95, 'C', fontsize=fontsize_panel_label, weight='bold')
    plt.savefig(os.path.join(figfolder, 'fig8.jpg'), dpi=300)
    plt.close()


def get_taxis_faxis(datafolder):
    file = glob.glob(os.path.join(datafolder, '*-fs20000.pkl'))[0]
    with open(file, 'rb') as f:
        session = pickle.load(f)
    unit = session.units[0]
    return unit.strf_taxis, unit.strf_faxis
