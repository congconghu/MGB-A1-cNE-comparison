# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:47:51 2022

@author: Congcong
"""
import re
import pickle
import os
import glob

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
import ne_toolbox as netools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = 'Arial'
figure_size = (18.3, 24.7)
activity_alpha = 99.5
colors = sns.color_palette("Paired")
A1_color = (colors[1], colors[0])
MGB_color = (colors[5], colors[4])


def plot_ne_split_ic_weight_match(ne_split, figpath):
    """
    plot matching patterns for each cNE under 3 conditions: cross condition, spon and dmr

    Inputs:
        ne_split: dictionary containing ne data on 4 blocks
        figpath: file path to save figures
    """
    colors = sns.color_palette("Paired")
    color_idx = [3, 2, 7, 6, 9, 8]
    colors = [colors[x] for x in color_idx]

    corr_mat = ne_split['corr_mat']
    dmr_first = ne_split['dmr_first']
    n_dmr = len(ne_split['order']['dmr'][0])
    n_spon = len(ne_split['order']['spon'][0])
    n_match = min([n_dmr, n_spon])

    for i in range(n_match):

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))

        if dmr_first:
            order = ne_split['order']['dmr'][1][i]
            ic_dmr = ne_split['dmr1'].patterns[order]
            order = ne_split['order']['spon'][0][i]
            ic_spon = ne_split['spon0'].patterns[order]
        else:
            order = ne_split['order']['dmr'][0][i]
            ic_dmr = ne_split['dmr0'].patterns[order]
            order = ne_split['order']['spon'][1][i]
            ic_spon = ne_split['spon1'].patterns[order]

        plot_matching_ic(axes[0], ic_dmr, ic_spon, colors[0], colors[2])
        order0 = ne_split['order']['dmr'][0][i]
        order1 = ne_split['order']['dmr'][1][i]
        plot_matching_ic(axes[1],
                         ne_split['dmr0'].patterns[order0],
                         ne_split['dmr1'].patterns[order1],
                         colors[0], colors[1])
        order0 = ne_split['order']['spon'][0][i]
        order1 = ne_split['order']['spon'][1][i]
        plot_matching_ic(axes[2],
                         ne_split['spon0'].patterns[order0],
                         ne_split['spon1'].patterns[order1],
                         colors[2], colors[3])

        plt.tight_layout()
        fig.savefig(re.sub('.jpg', '-{}.jpg'.format(i), figpath))
        plt.close()


def plot_matching_ic(ax1, ic1, ic2, color1, color2):
    """
    stem plot of matching patterns

    Input:
        ax1: axes to plot on
        ic1, ic2: cNE patterns
        color1, color2: color for the 2 patterns
    """

    markersize = 10
    # get ylimit for the plot
    ymax = max(ic1.max(), ic2.max()) * 1.1
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
    plt.setp(markerline, markersize=markersize, color=color1)
    plt.setp(stemline, color=color1)
    ax1.set_xlim([0, n_neuron + 1])
    ax1.set_ylim([-ymax, ymax])
    ax1.spines['left'].set_color(color1)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.tick_params(axis='y', colors=color1)

    # plot on the right axes
    markerline, stemline, baseline = ax2.stem(
        range(1, n_neuron + 1), ic2,
        markerfmt='o', basefmt='k')
    plt.setp(markerline, markersize=markersize, color=color2)
    plt.setp(stemline, color=color2)
    ax2.set_xlim([0, n_neuron + 1])
    ax2.set_ylim([-ymax, ymax])
    ax2.invert_yaxis()
    ax2.spines.top.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines['right'].set_color(color2)
    ax2.tick_params(axis='y', colors=color2)


def plot_ne_split_ic_weight_corr(ne_split, figpath):
    """
    heatmap of correlation values among matching patterns
    """
    corr_mat = ne_split['corr_mat']
    n_dmr = len(ne_split['order']['dmr'][0])
    n_spon = len(ne_split['order']['spon'][0])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(corr_mat, aspect='auto', cmap='Greys', vmin=0, vmax=1)
    plt.colorbar(im)
    # draw boundary for correlation matrix of dmr-evoked activities
    p = mpl.patches.Rectangle((-0.48, -0.48), n_dmr - .04, n_dmr - .04,
                              facecolor='none', edgecolor='green', linewidth=5)
    ax.add_patch(p)
    # draw boundary for correlation matrix of spontanoues activities
    p = mpl.patches.Rectangle((n_dmr - 0.48, n_dmr - 0.48), n_spon - .04, n_spon - .04,
                              facecolor='none', edgecolor='orange', linewidth=5)
    ax.add_patch(p)
    # draw boundary for correlation matrix of corss conditions
    if ne_split['dmr_first']:
        xy = (-0.48, n_dmr - 0.48)
        x, y = n_dmr - .04, n_spon - .04
    else:
        xy = (n_dmr - 0.48, -0.48)
        x, y = n_spon - .04, n_dmr - .04
    p = mpl.patches.Rectangle(xy, x, y,
                              facecolor='none', edgecolor='purple', linewidth=5)
    ax.add_patch(p)
    order = ne_split['order']['dmr'][0] + ne_split['order']['spon'][0]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    order = ne_split['order']['dmr'][1] + ne_split['order']['spon'][1]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    fig.savefig(figpath)
    plt.close(fig)


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


def plot_5ms_strf_ne_and_members(ne, savepath):
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
        plot_strf(ax, cne.strf, taxis=cne.strf_taxis, faxis=cne.strf_faxis)

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
            plot_strf(ax, unit.strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis)
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


def plot_raster(ax, units, offset='idx', color='k', new_order=[]):
    """raster plot of activities of member neurons"""
    for idx, unit in enumerate(units):
        if offset == 'idx':
            pass
        elif offset == 'unit':
            idx = unit.unit
        if any(new_order):
            idx = new_order[idx]
        ax.eventplot(unit.spiketimes, lineoffsets=idx + 1, linelengths=0.8, colors=color)


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
    xcorr = pd.read_json('E:\Congcong\Documents\data\comparison\data-summary\member_nonmember_pair_xcorr_filtered.json')
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


# ---------------------cNE properties ---------------------------------------------------
def num_ne_vs_num_neuron(ax, datafolder, stim):
    # get files for recordings in A1 and MGB
    files_all = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = {'A1': [], 'MGB': []}
    files['A1'] = [x for x in files_all if 'H22x32' in x]
    files['MGB'] = [x for x in files_all if 'H31x64' in x]

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
    files_all = glob.glob(os.path.join(datafolder, '*' + stim + '.pkl'))
    files = {'A1': [], 'MGB': []}
    files['A1'] = [x for x in files_all if 'H22x32' in x]
    files['MGB'] = [x for x in files_all if 'H31x64' in x]

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
                        linewidth=linewidth)
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


def plot_strf(ax, strf, taxis, faxis):
    """
    plot strf and format axis labels for strf

    Input
        ax: axes to plot on
        strf: matrix
        taxis: time axis for strf
        faxis: frequency axis for strf
    """
    max_val = abs(strf).max() * 1.01
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
