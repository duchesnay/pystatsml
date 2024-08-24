# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:58:31 2016

@author: edouard.duchesnay@cea.fr
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Ellipse

def plot_pvalue_under_h0(stat_vals, stat_probs, stat_obs, stat_h0=0,
                         bar_width=None,
                         thresh_low=None, thresh_high=None):
    """Plot p-value and observed statistic under null hypothesis

    Parameters
    ----------
    stat_vals : numpy array (1D)
        Statistic values under H0
    stat_probs : numpy array (1D)
        Probabilities of statistic (PDF or PMF) values under H0
    stat_obs : float
        Statistic observed
    stat_h0 : float
        Statistic under H0
    bar_width : None, float
        if bar_width is not None use bar plot, else use plot
    thresh_low : float
        Low threshold for observed statistic, low value (for two sided test)
    thresh_high : float
        Hight threshold for observed statistic, low value (for two sided test)
    """
    if bar_width is not None:
        plt.bar(stat_vals, stat_probs, color='k', width=bar_width, fill=False,
                label=r'$P(Stat|H_0)$')
    else:
        plt.plot(stat_vals, stat_probs, 'k-', label=r'$P(Stat|H_0)$')

    # p-value areas
    stat_probs_ = np.zeros(len(stat_probs))

    if thresh_low:
        stat_probs_[stat_vals <= thresh_low] = stat_probs[stat_vals <= thresh_low]
    if thresh_high:
        stat_probs_[stat_vals >= thresh_high] = stat_probs[stat_vals >= thresh_high]   

    dx = np.diff(stat_vals)[0]
    plt.bar(stat_vals, stat_probs_, color="#1f77b4", width=dx, label=r'p-value')

    #plt.fill_between(stat_vals, 0, stat_probs_,
    #                color="#1f77b4", alpha=.8, label="p-value", step='post')
    plt.axvline(x=stat_obs, color='r', ls='--', lw=2, label='Stat. observed')
    plt.axvline(x=stat_h0, color="k", lw=2, label=r'Stat. H0')

    plt.legend()
    plt.title('Observed statistic under null hypothesis')

def plot_pvalue_under_h0_(stat_vals, stat_probs, stat_obs, thresh_low=None, thresh_high=None):
    """Plot p-value and observed statistic under null hypothesis

    Parameters
    ----------
    stat_vals : numpy array (1D)
        Statistic values under H0
    stat_probs : numpy array (1D)
        Probabilities of statistic (PDF or PMF) values under H0
    stat_obs : float
        Observed statistic   
    thresh_low : float
        Low threshold for observed statistic, low value (for two sided test)
    thresh_high : float
        Hight threshold for observed statistic, low value (for two sided test)
    """

    plt.plot(stat_vals, stat_probs, 'k-', label="$P(Stat|H_0)$")

    # p-value areas
    stat_probs_ = np.zeros(len(stat_probs))

    if thresh_low:
        stat_probs_[stat_vals <= thresh_low] = stat_probs[stat_vals <= thresh_low]
    if thresh_high:
        stat_probs_[stat_vals >= thresh_high] = stat_probs[stat_vals >= thresh_high]   

    plt.fill_between(stat_vals, 0, stat_probs_,
                    alpha=.8, label="p-value", step='post')
    plt.axvline(x=stat_obs, color='r', ls='--', lw=2, label='Observed Stat.')

    plt.legend()
    plt.title('Observed statistic under null hypothesis')

 
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
