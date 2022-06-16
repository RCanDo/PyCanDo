# -*- coding: utf-8 -*-
#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Helper plot functions
version: 1.0
type: sub-module
keywords: [plot, ]
description: |
content:
    -
remarks:
todo:
sources:
file:
    usage:
        interactive: False
        terminal: False
    date: 2021-10-30
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

#%% imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import math as m

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #, LinearSegmentedColormap
#import mpl_toolkits as mplt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import gaussian_kde
import statsmodels.api as sm

from sklearn.metrics import roc_curve, roc_auc_score

from utils.builtin import flatten, coalesce
from utils.df import info, summary
from utils.transformations import power_transformer

#%%
def distribution(variable):
    """
    variable  list/pd.n/ series of floats/integers
        interpreted as n samples of one numeric variable
    """
    n = len(variable)

    variable = variable.sort_values().to_numpy()

    return variable, np.arange(n+1)[1:]/n  # x, F(x) - distribution of variable

#%%
def agg_for_bins(variable, bins=None, agg=sum):
    """
    variable  list/series of floats/integers
        interpreted as n samples of one numeric variable
    """
    variable.dropna(inplace=True)

    if bins is None:
        bins = 5

    if isinstance(bins, int):
        bins = np.linspace(min(variable), max(variable), bins + 1)

    rng = max(variable) - min(variable)

    bins[0] -= rng*.01

    aggs = [agg(variable.loc[ (variable > bins[k]) & (variable <= bins[k+1]) ]) for k in range(len(bins) - 1)]

    return aggs, bins   # list of agg values for each bin, list of bins borders

#%%
def to_factor(x, as_factor=None, factor_threshold=13,
              factor_types=["category", "object", "str"],
              most_common=13,   #!!!???
              ):
    """
    Determining if the `x` is factor or not;
    if determined as factor its dtype is turned to 'category'
    Arguments
    ---------
    x : pd.Series
    as_factor : None; bool
        if not None then this is the value returned,
        i.e. x is forced to be factor or not;
    factor_threshold : 13; int
        if `x` has less then `factor_threshold` unique values
        then it is assessed as factor;
    factor_types : List[str]
        which data types consider as factor types
    most_common : int
        how many of the most common levels we want to consider
        later on (e.g. in plot_covariates()) to be taken into consideration;
        this limits the size of the value counts table:
        `x.value_counts()[:most_common]` see (*) in the code;
        notice though that the `x` is returned whole! -- NOT only most_common levels;
        !!! this rather should not be part of this fun !!!
    Returns
    -------
    as_factor: bool
    x: pd.Series
    x_vc: pd.Series
    """
    if as_factor is None:
        as_factor = x.dtype in factor_types
        if not as_factor:
            as_factor = x.unique().shape[0] < factor_threshold

    if as_factor:
        #x = x.astype(str).astype("category")                        #!!!
        x = x.astype("category")                        #!!!

        ##-----------------------------------------------------
        x_vc = x.value_counts()
        n_levels = len(x_vc)

        if most_common and most_common < n_levels:
            x_vc = x_vc.iloc[:most_common]             # (*)
            #! x = x[x.isin(x_vc.index)]  #!!! NO !!! we want everything!

        #else:
        #    most_common = n_levels

    else:
        x_vc = None

    return as_factor, x, x_vc  # bool,  (factorised) variable,  variable.value_counts()

#%%
def make_title(varname, lower=None, upper=None, transform=False, lower_t=None, upper_t=None, tex=False):
    """
    transname(varname [lower, upper]) [lowetr_t, upper_t]
    """
    if not tex:
        varname = varname.replace("_", "\\_")

    if lower or upper:
        lower = f"[{lower}, " if lower else "(-\\infty, "
        upper = f"{upper}]" if upper else "\\infty)"
        lims = lower + upper
        #
        title = f"{varname}_{{{lims}}}"
    else:
        title = varname

    if isinstance(transform, str):
        transname = transform
    else:
        if transform is None:
            transname = "T"
        else:
            if transform:
                try:
                    transname = coalesce(transform.__name__, "T")
                except:
                    transname = "T"
            else:
                transname = False

    if transname:    # maybe False -- no transformation
        if lower_t or upper_t:
            lower_t = f"[{lower_t}, " if lower_t else "(-\\infty, "
            upper_t = f"{upper_t}]" if upper_t else "\\infty)"
            lims_t = lower_t + upper_t
            #
            title = f"{transname}\\left(\\ {title}\\ \\right)_{{{lims_t}}}"
        else:
            title = f"{transname}\\left(\\ {title}\\ \\right)"

    return f"${title}$"

#%%
def style_affairs(style, color, grid, suptitlecolor, titlecolor, alpha, N):
    """"""
    if style:
        if isinstance(style, str):
            plt.style.use(style)
        else:
            pass
            # use graphic params set externally
    else:
        style = 'dark_background'
        plt.style.use(style)

    if not color:
        if mpl.rcParams['axes.facecolor'] == 'black':
            color = 'y'
        else:
            color = 'k'

    if grid:
        mpl.rc('axes', grid=True)
        if isinstance(grid, bool):
            if mpl.rcParams['axes.facecolor'] == 'black':
                grid = dict(color='darkgray', alpha=.3)

    if titlecolor is None:
        if mpl.rcParams['axes.facecolor'] == 'black':
            mpl.rc('axes', titlecolor='gray')
        else:
            mpl.rc('axes', titlecolor='dimgray')

    if suptitlecolor is None:
        if mpl.rcParams['axes.facecolor'] == 'black':
            suptitlecolor='lightgray'
        else:
            suptitlecolor='k'

    if alpha is None:
        #N = len(variable)
        a = 0.00023   # = m.log(10)/(1e4 - 1)
        alpha = m.exp(-a*(N - 1))

    return  style, color, grid, suptitlecolor, titlecolor, alpha

#%%
def set_xscale(ax, scale):
    """setting the scale type on x axis
    scale may be str or tuple or dict
    each is passed to ax.set_scale() and unpacked (except str)
    hence the elements of `scale` must be as described in
    https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.axes.Axes.set_xscale.html?highlight=set_xscale#matplotlib.axes.Axes.set_xscale
    https://matplotlib.org/3.5.1/api/scale_api.html#matplotlib.scale.ScaleBase
    etc.
    example:
    https://matplotlib.org/3.5.1/gallery/scales/scales.html#sphx-glr-gallery-scales-scales-py
    """
    if isinstance(scale, str):
        ax.set_xscale(scale)
    elif isinstance(scale, tuple):
        ax.set_xscale(*scale)
    elif isinstance(scale, dict):
        ax.set_xscale(**scale)

#%%
def set_yscale(ax, scale):
    """setting the scale type on y axis
    scale may be str or tuple or dict
    each is passed to ax.set_scale() and unpacked (except str)
    ... see set_xscale()
    """
    if isinstance(scale, str):
        ax.set_yscale(scale)
    elif isinstance(scale, tuple):
        ax.set_yscale(*scale)
    elif isinstance(scale, dict):
        ax.set_yscale(**scale)


#%%
def set_grid(ax, off="both", grid=None):
    """
    off : "both" / "x" / "y"
        axis to be always turned off if not stated otherwise
    grid : False; bool or dict;
        if False then no grid is plotted (regardless of style);
        if True then grid is plotted as ascribed to given style;   !!! some styles do not print grid !
        in case of "black_background" it is dict(color='gray', alpha=.3)
        if dict then values from this dict will be applied as described in
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.grid.html
    """

    if grid:
        if not isinstance(grid, bool):
            ax.grid(**grid)
    else:
        second_axis = {"x": "y", "y": "x", "both": "both"}[off]
        ax.grid(visible=False, axis=second_axis)

    if off != "both":
        ax.grid(visible=False, axis=off)

#%%
def set_title(ax, title, color):
    """this is for axis title
    """
    if color:
        ax.set_title(title, color=color)
    else:
        ax.set_title(title)

#%%
## ROC
def roc(binary, score):
    """
    binary : pd.Series / np.array
        binary variable. i.e. in {0, 1} or {-1, 1}
    score : pd.Series / np.array
        score on binary i.e. model probablity of getting 1
    """

    fpr, tpr, thresholds = roc_curve(binary, score)#, pos_label=2)
    roc_auc = roc_auc_score(binary, score)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


#%%
