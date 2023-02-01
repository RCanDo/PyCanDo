#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Helper plot functions
version: 1.0
type: sub-module
keywords: [plot, preprocessing]
description: |
content:
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

# %% imports
import warnings
warnings.filterwarnings('ignore')

from typing import Union, Tuple, Iterable

import numpy as np
import pandas as pd
import math as m
# import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from matplotlib.colors import ListedColormap  #, LinearSegmentedColormap
# import mpl_toolkits as mplt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from scipy.stats import gaussian_kde
# import statsmodels.api as sm

from sklearn.metrics import roc_curve, roc_auc_score

from common.builtin import flatten, coalesce, union
from common.transformations import power_transformer

from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# %%
default_cycler = [
    (0.0, '#17becf'),  # tab:cyan  (~teal)
    (0.1, '#2ca02c'),  # tab:green
    (0.2, '#7f7f7f'),  # tab:grey  (medium dark)
    (0.35, '#ff7f0e'),  # tab:orange
    (0.5, '#bcbd22'),  # tab:olive (green-yellow-grey)
    (0.65, '#d62728'),  # tab:red
    (0.75, '#9467bd'),  # tab:purple (violet)
    # '#8c564b',  # tab:brown   # too dark
    (0.85, '#e377c2'),  # tab:pink
    (1.0, '#1f77b4'), ]  # tab:blue

mpl.cm.register_cmap("ak01", LinearSegmentedColormap.from_list("ak01", default_cycler))


# %%
def get_cmap(cmap: str, n: int):
    cmap = mpl.cm.get_cmap(cmap, n)
    if not hasattr(cmap, "colors"):
        cmap = ListedColormap(cmap(np.linspace(0., 1., n))[:, :3])
    return cmap


# %%
def brightness(p, l):
    """luminosity `l` of a given `color`
    p : float in [0, 1];
    l : float >= 0;
        l = 1 does not change p;
        l > 1 makes p "brighter" i.e. inceases it's value:  p' = p*a + (1 - a)  where  a = 1/l;
        l in [0, 1] makes p 'darker' i.e. lowers it's value:  p' = p*l;
    Notice that this funciton is intended to work on np.arrays[n, 3]
    where each row is a tuple of length 3 with values of each RGB channel given as p in [0, 1];
    E.g.
    cat_colors = plt.colormaps['gist_ncar'](np.linspace(0.2, 0.8, N))[:, :3]   # we don't want alpha channel here
    cat_colors_brighter = brightness(cat_colors, 2)
    cmap = ListedColormap(cat_colors_brighter)    # final cmap (if one needs to use cmap)
    """
    if l > 1.:
        l = 1 / l
        p = p * l + (1 - l)
    elif l >= 0:
        p = p * l
    else:
        raise Exception("`l` must be >= 0")
    return p


# %%
def is_mpl_color(color: str):
    """checking if given name is valid matplotlib color name"""
    res = color in union(
        mpl.colors.BASE_COLORS.keys(),
        mpl.colors.TABLEAU_COLORS.keys(),
        mpl.colors.CSS4_COLORS.keys(),
        mpl.colors.XKCD_COLORS.keys())
    return res


# %%
def image(file='img.png', dpi=100):
    """
    dpi : int; dots per inch
    """
    img = mpimg.imread(file)
    h, w = img.shape[:2]      # size in pixels
    # hence  h/dpi, w/dp  is size in inches  IF  dpi is right
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi))
    ax.imshow(img)
    ax.grid(visible=False)
    ax.axis('off')


# %%
def get_var_and_name(
        variable: Union[str, Iterable],
        data: pd.DataFrame,
        varname: str = None,
        default_name: str = "variable", ) -> Tuple[pd.Series, str]:
    """"""
    if isinstance(variable, str):
        variable = data[variable].copy()
    else:
        variable = pd.Series(variable)      # for lists or np.arrays
        if variable.name is None:
            variable.name = coalesce(varname, default_name)

    if varname is None:
        varname = coalesce(variable.name, default_name)

    return variable, varname


# %%
def distribution(variable):
    """
    variable  list/pd.n/ series of floats/integers
        interpreted as n samples of one numeric variable
    """
    n = len(variable)

    variable = variable.sort_values().to_numpy()

    return variable, np.arange(n + 1)[1:] / n  # x, F(x) - distribution of variable


# %%
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

    bins[0] -= rng * .01

    aggs = [agg(variable.loc[(variable > bins[k]) & (variable <= bins[k + 1])]) for k in range(len(bins) - 1)]

    return aggs, bins   # list of agg values for each bin, list of bins borders


# %%  exactly the same in df.helpers
def sample(
        data: Union[pd.DataFrame, pd.Series],
        n: int, shuffle: bool, random_state: int) -> Union[pd.DataFrame, pd.Series]:
    """"""
    if n and n < len(data):
        data = data.sample(int(n), ignore_index=False, random_state=random_state)
    if shuffle:
        data = data.sample(frac=1, ignore_index=True, random_state=random_state)
    else:
        data = data.sort_index()  # it is usually original order (but not for sure...)
    return data


# %%
def clip_transform(
        x,
        lower=None, upper=None, exclude=None,
        transformation=None,
        lower_t=None, upper_t=None, exclude_t=None, transname="T"):
    """
    clipping -> transformng -> clipping
    """
    if lower is not None:
        x = x[x >= lower]
    if upper is not None:
        x = x[x <= upper]
    if exclude is not None:
        x = x[~ x.isin(flatten([exclude]))]

    if transformation:

        if isinstance(transformation, bool):
            x, transformation = power_transformer(x)
        else:
            x = pd.Series(transformation(x))

        transname = coalesce(transformation.__name__, transname)

        if lower_t is not None:
            x = x[x >= lower_t]
        if upper_t is not None:
            x = x[x <= upper_t]
        if exclude_t is not None:
            x = x[~ x.isin(flatten([exclude_t]))]
    else:
        transname = None

    return x, transname


# %%
def to_datetime(t):
    """
    as pd.to_datetime but returns None instead of NaT (numpy.datetime64('NaT'))
    """
    t = pd.to_datetime(t) if t else None
    t = None if str(t) == 'NaT' else t
    return t


# %%
def to_factor(x, as_factor=None, factor_threshold=13,
              most_common=13,   # !!!???
              factor_types=["category", "object", "str"], ):
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
        x = x.astype("category")                        # !!!

        x_vc = x.value_counts()
        n_levels = len(x_vc)

        if most_common and most_common < n_levels:
            x_vc = x_vc.iloc[:most_common]             # (*)
            # x = x[x.isin(x_vc.index)]  # !!! NO !!! we want everything!

    else:
        x_vc = None

    return as_factor, x, x_vc   # bool,  (factorised) variable,  variable.value_counts()


# %%
def datetime_to_str(dt):
    """"""
    if dt is not None:

        dt = pd.to_datetime(dt)

        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            dt = dt.strftime("%Y.%m.%d")
        elif dt.minute == 0 and dt.second == 0:
            dt = dt.strftime("%Y.%m.%d / %H")
        elif dt.second == 0:
            dt = dt.strftime("%Y.%m.%d / %H:%M")
        else:
            dt = dt.strftime("%Y.%m.%d / %H:%M:%S")

    return dt


# %%
def make_title(varname, lower=None, upper=None, transname=None, lower_t=None, upper_t=None, tex=False):
    """
    transname(varname_[lower, upper])_[lowetr_t, upper_t]

    varname : str
    lower : str, numeric, None;
    upper : str, numeric, None;
        lower/upper limit before transformation (if any)
    transname : str, None;
        if str this string is taken as transformation name;
        if None then it's assumed that no transformation is done;
           then lower_t and upper_t are ignored;
    lower_t : str, numeric, None;
    upper_t : str, numeric, None;
        lower/upper limit after transformation (if any)
    """
    if isinstance(varname, tuple):      # in case of MultiIndex
        varname = ".".join(varname)

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

    if transname:
        if lower_t or upper_t:
            lower_t = f"[{lower_t}, " if lower_t else "(-\\infty, "
            upper_t = f"{upper_t}]" if upper_t else "\\infty)"
            lims_t = lower_t + upper_t
            #
            title = f"{transname}\\left(\\ {title}\\ \\right)_{{{lims_t}}}"
        else:
            title = f"{transname}\\left(\\ {title}\\ \\right)"

    return f"${title}$"


# %%
def style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha, N):
    """"""
    DARK_COLORS = ['black', 'rarkgray', 'darkgrey']

    if style:
        if isinstance(style, str):
            plt.style.use(style)
        else:
            pass
            # use graphic params set externally
    else:
        style = 'dark_background'
        plt.style.use(style)

    if mpl.rcParams['axes.facecolor'] in DARK_COLORS:
        brightness = 2.5 if brightness is None else brightness
    else:
        brightness = 1 if brightness is None else brightness

    if color is None:
        if mpl.rcParams['axes.facecolor'] in DARK_COLORS:
            color = 'y'
        else:
            color = 'k'

    if grid:
        mpl.rc('axes', grid=True)
        if isinstance(grid, bool):
            if mpl.rcParams['axes.facecolor'] in DARK_COLORS:
                grid = dict(color='darkgray', alpha=.3)

    if axescolor is None:
        if mpl.rcParams['axes.facecolor'] in DARK_COLORS:
            axescolor = 'dimgray'
            mpl.rc('axes', edgecolor=axescolor)
        else:
            axescolor = 'gray'
            mpl.rc('axes', edgecolor=axescolor)

    if titlecolor is None:
        if mpl.rcParams['axes.facecolor'] in DARK_COLORS:
            titlecolor = 'gray'
            mpl.rc('axes', titlecolor=titlecolor)
        else:
            titlecolor = 'dimgray'
            mpl.rc('axes', titlecolor=titlecolor)

    if suptitlecolor is None:
        if mpl.rcParams['axes.facecolor'] in DARK_COLORS:
            suptitlecolor = 'lightgray'
        else:
            suptitlecolor = 'k'

    if alpha is None:
        # N = len(variable)
        a = 0.00023   # = m.log(10)/(1e4 - 1)
        alpha = max(m.exp(-a * (N - 1)), .05)

    return style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha


# %%
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


# %%
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


# %%
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


# %%
def set_title(ax, title, color):
    """this is for axis (subplot) title
    """
    if title:
        if color:
            ax.set_title(title, color=color)
        else:
            ax.set_title(title)


def set_figtitle(fig, title, suptitlecolor, suptitlesize, fontweight='normal'):
    """this is for the main figure title
    """
    if title:
        if suptitlecolor:
            fig.suptitle(title, fontweight=fontweight, color=suptitlecolor, fontsize=15 * suptitlesize)
        else:
            fig.suptitle(title, fontweight='normal', fontsize=15 * suptitlesize)


# %%
def set_axescolor(ax, color):
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)


# %%
# ROC
def roc(binary, score):
    """
    binary : pd.Series / np.array
        binary variable. i.e. in {0, 1} or {-1, 1}
    score : pd.Series / np.array
        score on binary i.e. model probablity of getting 1
    """

    fpr, tpr, thresholds = roc_curve(binary, score)  # , pos_label=2)
    roc_auc = roc_auc_score(binary, score)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc, )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


# %%
