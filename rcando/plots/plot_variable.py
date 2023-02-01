#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Diagnostic plots for one variable
version: 1.0
type: module
keywords: [plot, ]
description: |
    Custom diagnostic plots for one variable;
    For numeric:
        - histogram
        - cloud
        - density
        - distribution
        - sum vs counts (wrt to groups from histogram)
        - boxplot
    or just:
        - barplot
    for categorical.
    Any configuration of the above types of plots are possible via `what` parameter.
    The idea is to make it automated wrt different types of variables
    (numeric / categorical);
    maximum flexibility (lots of parameters) but with sensible defaults.
    This allows to do well with difficult cases like numeric variables with
    small nr of different values (better to plot it as categorical)
    or categorical variables with large number of different values
    (better to (bar)plot only most common values), etc.
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

# %%
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# import matplotlib as mpl
import matplotlib.pyplot as plt

from common.builtin import coalesce
import common.df as cdf
import common.plots.helpers as h


# %%
def plot_numeric(
        variable, data=None,
        what=[["hist", "cloud", "density"], ["agg", "boxplot", "distr"]],
        varname=None, title=None, title_suffix=None,
        # Variable modifications (before plotting)
        lower=None, upper=None, exclude=None,
        transform=False,
        upper_t=None, lower_t=None, exclude_t=None,
        #
        bins=7, agg=sum,
        n_obs=int(1e4), shuffle=False, random_state=None,
        # Graphical parameters
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=5, width_adjust=1.2,  # for the single plot
        scale="linear",
        lines=True,
        cmap="Set2",  # for coloring of bars in "hist" and respective points of "agg"
        color=None, s=9, alpha=None, brightness=None,  # alpha, size and brightness of a data point in a "cloud"
        ignore_index=False,
        style=True, grid=True, axescolor=None, titlecolor=None,
        suptitlecolor=None, suptitlesize=1.,  # multiplier of 15
        #
        print_info=True, res=False,
        *args, **kwargs):
    """
    Remarks:
        - `style` is True by default what means using style set up externally
          and it is assumed to be set to  plt.style.use('dark_background');
        - All default graphic parameters are set for best fit
          with 'dark_background' style.
        - Unfortunately changing styles is not fully reversible, i.e.
          some parameters (plt.rcParams) stays changed after reverting style;
          (eg. grid stays white after 'ggplot',
          the same for title colors after 'black_background', etc);
          Just because of this there are some parameters like `color`, `grid`, `titlecolor`
          to set up their values by hand in case of unwanted "defaults".

    Basic params
    ------------
    variable : str or pd.Series;
        if str then it indicates column of `data`;
        else pd.Series of data to be plotted;
    data : None; pd.DataFrame;
        if None then `variable` must be pd.Series with data in interest;
    what : [['hist', 'cloud'], ['boxplot', 'density'], ['agg', 'distr']]; list (of lists);
        the whole list reflects the design of the whole final figure where
        each sublist represents one row of plots (axes) within a figure
        and each element of a sublist is the name of the respective plot type
        which will be rendered in a respective subplot (axis) of the figure;
        thus each sublist should be of the same length however...
        possible values of the elements of the sublist (types of the plots) are:
            "hist", "cloud", "dist", "density", "agg", "boxplot", "blank" (for empty subplot);
    varname : None; str;
        variable name to be used in title, etc.; if None then taken from
        .name attr of `variable`;
    title : None; str;
        title of a plot; if None then generated automaticaly;

    Variable modifications (before plotting)
    ----------------------------------------
    lower : numeric or None;
        lower limit of`variable` to be plotted; inclusive !
        if None then `lower == min(variable)`
    upper : numeric or None;
        upper limit of`variable` to be plotted; inclusive !
        if None then `upper == max(variable)`
    exclude : numeric or list of numerics;
        values to be excluded from `variable` before plotting;
    transform : None or bool or function;
        if None or False no transformation is used;
        if True then Yeo-Johnson transformation is used with automatic parameter;
        if function is passed then this function is used;
    upper_t : numeric or None;
        upper limit of transformed `variable` to be plotted; inclusive !
        if None then `upper == max(variable)`
    lower_t : numeric or None;
        lower limit of transformed `variable` to be plotted; inclusive !
        if None then `lower == min(variable)`
    exclude_t : numeric or list of numerics;
        values to be excluded from transformed `variable` before plotting;
    agg : function;
        type of aggregate for "agg" plot where for each group aqured from "hist"
        we plot point (having the same color as respective bar of "hist")
        with coordinates (count, agg) where `count` is nr of elements in a group
        and `agg` is aggregate of values for this group.
    bins : int or list of boarders of bins (passed to ax.hist(...))
        how many or what bins (groups) for "hist" and "agg";
    n_obs : int(1e4); int or None;
        if not None then maximum nr of observations to be sampled from variable before plotting
        'cloud', 'density', 'distr';
        if None whole data will be plotted (what is usually not sensible for very large data).
    shuffle : False (boolean);
        shuffle data before plotting -- useful only for "cloud" plot in case
        when data are provided in clumps with different statistical properties;
        shuffling helps to spot distribution features common to the whole data.
    random_state : None; int;
        passed to numpy random generator for reproducibility in case of
        `n_obs` is not None or shuffle is True;

    Graphical parameters
    --------------------

    #  Sizes for the whole figure
        These params overwrite single-plot-sizes params.
    figsize : None; tuple of numerics (figwidth, figheight)
    figwidth : None; numeric
    figheight : None; numeric

    #  Sizes for the single plot
        If width and height are None they are
    width : None; numeric
        = size if is None
    height : None; numeric
        = size * width_adjust if is None
    size : 4; numeric
        may be None only if width and height are not None or fig-sizes params are not None
    width_adjust : 1.2; numeric
        if width not set up directly then `width = size * width_adjust`
    scale : "linear"
    lines : True; boolean
    cmap : "ak01";
        color map name;
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py
        or dir(matplotlib.pyplot.cm) for list of all available color maps;
        see https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#creating-colormaps-in-matplotlib
        on how to create and register ListedColormaps.
    alpha : None; float between 0 and 1
        for points of "cloud" only;
    s : .1; float;
        size of a data point in a "cloud"
    style : True; bool or str
        if True takes all the graphic parameters set externally (uses style from environment);
        if False then is set to "dark_background";
        str must be a name of one of available styles: see `plt.style.available`.
    color : None; str
        color of lines and points for 'cloud', 'boxplot', 'density' and 'distr';
        if None then set to "yellow" for style "black_background", else to "black";
    grid : False; bool or dict;
        if False then no grid is plotted (regrdless of style);
        if True then grid is plotted as ascribed to given style;
        in case of "black_background" it is dict(color='gray', alpha=.3)
        if dict then values from this dict will be applied as described in
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.grid.html
        Most general form (example values):
        { 'alpha': 1.0,  'color': '#b0b0b0',  'linestyle': '-',  'linewidth': 0.8  }
    titlecolor : None; str
        color of axis titles;
    suptitlecolor : None; str
        color of the whole title plot (fig.suptitle)
    suptitlesize : 1.; float
        multiplier of 15 for the whole title plot (fig.suptitle)

    print_info : True
        print df.info(variable) (after all transformations)
    ret : False
        do return result of all the calculations?
        default is False and then None is returned;
        otherwise (if True) dictionary is returned with the following structure:
        { "plot_type_name": results of calculations for this plot,
          ...
        }

    Returns
    -------
    dictionary of everything...
    """

    # -------------------------------------------------------------------------
    #  loading data

    variable, varname = h.get_var_and_name(variable, data, varname, "X")

    # !!! index is ABSOLUTELY CRITICAL here !!!
    if ignore_index:
        variable, color, s, alpha = cdf.align_indices(variable, color, s, alpha)

    # -----------------------------------------------------
    #  info on raw variable
    var_info = cdf.info(pd.DataFrame(variable), what=["dtype", "oks", "oks_ratio", "nans_ratio", "nans", "uniques"])

    if print_info:
        print(" 1. info on raw variable")
        print(var_info)

    # -------------------------------------------------------------------------
    #  preparing data

    variable = variable.dropna()

    # -----------------------------------------------------
    #  transformation and clipping

    variable, transname = h.clip_transform(
        variable,
        upper, lower, exclude,
        transform, upper_t, lower_t, exclude_t, "T")

    # -----------------------------------------------------
    #  statistics for processed variable

    var_variation = cdf.summary(
        pd.DataFrame(variable),
        what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    var_distribution = cdf.summary(
        pd.DataFrame(variable),
        what=["range", "iqr", "mean", "median", "min", "max", "negatives", "zeros", "positives"])

    if print_info:
        print()
        print(" 2. statistics for processed variable")
        print(var_variation)
        print()
        print(var_distribution)

    # -----------------------------------------------------
    #  title

    if not title:
        title = h.make_title(varname, lower, upper, transname, lower_t, upper_t)

    if title_suffix:
        title = title + title_suffix

    # -----------------------------------------------------

    counts = None
    aggs = None

    # ----------------------------------------------------
    # !!! result !!!

    result = {
        "title": title,
        "variable": variable,    # processed
        "info": var_info,
        "variation": var_variation,     # variable after all prunings and transformations
        "distribution": var_distribution,
        "plot": dict()}

    # ---------------------------------------------------------------------------------------------
    #  plotting

    len_bins = len(bins) - 1 if isinstance(bins, list) else bins
    cmap = h.get_cmap(cmap, len_bins)

    # -------------------------------------------------------------------------
    #  style affairs

    N = len(variable) if not n_obs else min(len(variable), int(n_obs))

    # !!! get  color, s, alhpa  from data if they are proper column names !!!

    if isinstance(alpha, str):
        alpha = data[alpha]

    if isinstance(s, str):
        s = data[s]

    # take color from data only if it's not a color name
    if isinstance(color, str) and not h.is_mpl_color(color) and color in data.columns:
        color = data[color]

    color_data = color
    if not isinstance(color, str) and isinstance(color, Iterable):
        color = None

    style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha = \
        h.style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha, N)

    if color_data is None:
        color_data = color

    # -------------------------------------------------------------------------
    #  plot types

    def hist(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        #  ---------
        nonlocal bins
        nonlocal counts
        counts, bins, patches = ax.hist(variable, bins=bins)
        for p, c in zip(patches.patches, cmap.colors):
            p.set_color(c)
        #  ---------
        # ax.set_xscale(scale)                  # ???
        h.set_grid(ax, off="x", grid=grid)
        # set_axescolor(ax, axescolor)
        #
        result = dict(counts=counts, bins=bins, patches=patches)
        return dict(ax=ax, result=result)

    def agg_vs_count(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        #  ---------
        nonlocal bins
        nonlocal counts
        nonlocal aggs
        if counts is None:
            counts, bins = np.histogram(variable, bins=bins)
        aggs, bins = h.agg_for_bins(variable, bins, agg)
        scatter = ax.scatter(
            counts, aggs,
            s=50, color=cmap.colors, marker="D")
        #  ---------
        h.set_xscale(ax, scale)
        h.set_yscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        # set_axescolor(ax, axescolor)
        #
        result = dict(aggs=aggs, bins=bins, scatter=scatter)
        return dict(ax=ax, result=result)

    def boxplot(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        # ---------
        result = ax.boxplot(
            variable,
            vert=False,
            notch=True,
            #
            patch_artist=True,                              # !!!
            boxprops=dict(color=color, facecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color, marker="|"),
            medianprops=dict(color='gray' if color in ['k', 'black'] else 'k'),
            #
            showmeans=True,
            # meanline=False,
            meanprops=dict(  # color='white' if color in ['k', 'black'] else 'k',
                             marker="d",
                             markeredgecolor=color,
                             markerfacecolor='white' if color in ['k', 'black'] else 'k', markersize=17))
        # ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="y", grid=grid)
        # set_axescolor(ax, axescolor)
        #
        return dict(ax=ax, result=result)

    def density(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        # ---------
        try:
            density = gaussian_kde(variable.astype(float))
        except Exception:
            density = gaussian_kde(variable)
        xx = np.linspace(min(variable), max(variable), 200)
        lines = ax.plot(xx, density(xx), color=color)  # list of `.Line2D`
        # ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        # set_axescolor(ax, axescolor)
        #
        result = dict(xx=xx, lines=lines)
        return dict(ax=ax, result=result)

    def cloud(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        # ---------
        result = ax.scatter(variable, range(len(variable)), s=s, color=color_data, alpha=alpha)
        # ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        # set_axescolor(ax, axescolor)
        #
        return dict(ax=ax, result=result)

    def distr(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        #  ---------
        # # line version
        # result = ax.plot(*h.distribution(variable), color=color, linewidth=1)
        # dots version
        result = ax.scatter(*h.distribution(variable), s=.1, color=color_data)
        # `~matplotlib.collections.PathCollection`
        #  ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        # set_axescolor(ax, axescolor)
        #
        return dict(ax=ax, result=result)

    def blank(ax, title="", text="", *args, **kwargs):
        """"""
        h.set_title(ax, title, titlecolor)
        #  ---------
        ax.plot()
        ax.axis('off')
        ax.text(
            0.5, 0.5, text,
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            color='gray', fontsize=10)
        return dict(ax=ax, result=False)

    def error(ax, title=None):
        """"""
        h.set_title(ax, title, titlecolor)
        #  --------
        ax.plot()
        ax.axis('off')
        ax.text(
            0.5, 0.5, 'unavailable',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            color='gray', fontsize=10)
        return dict(ax=ax, result=False)

    PLOTS = {
        "hist": {"plot": hist, "name": "histogram"},
        "boxplot": {"plot": boxplot, "name": "box-plot"},
        "agg": {"plot": agg_vs_count, "name": f"{agg.__name__} vs count"},
        "cloud": {"plot": cloud, "name": "cloud"},
        "density": {"plot": density, "name": "density"},
        "distr": {"plot": distr, "name": "distribution"},
        "blank": {"plot": blank, "name": ""},
        "error": {"plot": error, "name": "error"},
    }

    # ------------------------------------------------------------------------
    #  plotting procedure

    # -----------------------------------------------------
    #  figure and plots sizes
    what = np.array(what, ndmin=2)
    nrows = what.shape[0]
    ncols = what.shape[1]

    if figsize is None:

        if figheight is None:
            height = size if height is None else height
            figheight = height * nrows + 1     # ? +1 ?

        if figwidth is None:
            width = size * width_adjust if width is None else width
            figwidth = width * ncols

        figsize = figwidth, figheight

    # ----------------------------------------------------
    #  core

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = np.reshape(axs, (nrows, ncols))    # unfortunately it's necessary because ...

    for t in ["hist", "boxplot", "agg", "blank"]:
        if t in what:
            ax = axs[np.nonzero(what == t)][0]
            try:
                result['plot'][t] = PLOTS[t]["plot"](ax, PLOTS[t]["name"])
            except Exception as e:
                print(e)
                result['plot'][t] = PLOTS["error"]["plot"](ax, PLOTS[t]["name"])

    variable = h.sample(variable, n_obs, shuffle, random_state)
    variable, color, s, alpha, color_data = \
        cdf.align_nonas(variable, color=color, s=s, alpha=alpha, color_data=color_data)

    for t in ["cloud", "density", "distr"]:
        if t in what:
            ax = axs[np.nonzero(what == t)][0]
            try:
                result['plot'][t] = PLOTS[t]["plot"](ax, PLOTS[t]["name"])
                if lines and not isinstance(bins, int):
                    for l, c in zip(bins, np.vstack([cmap.colors, cmap.colors[-1]])):
                        ax.axvline(l, color=c, alpha=.3)
            except Exception as e:
                print(e)
                result['plot'][t] = PLOTS["error"]["plot"](ax, PLOTS[t]["name"])

    result['plot']['axs'] = axs

    # -------------------------------------------------------------------------
    #  final

    if print_info:
        print()
        if isinstance(bins, Iterable):
            print("  For histogram groups:")
            #
            print("bins: [", end="")
            print(", ".join(f"{b:.2g}" for b in bins), end="")
            print("]")
        #
        if aggs:
            print(f"counts: {counts}")
            aggs_rounded = [round(a) for a in aggs]
            print(f"{agg.__name__}: {aggs_rounded}")

    h.set_figtitle(fig, title, suptitlecolor, suptitlesize)

    fig.tight_layout()
    # plt.show()

    result['plot']["fig"] = fig

    return None if not res else result


# %%
def plot_factor(
        variable, data=None, varname=None, title=None, title_suffix=None,
        most_common=13, print_levels=False,  # prints all levels regardless of `most_common`
        sort_levels=False, ascending=None,  # adopts to `sort_levels`
        dropna=False,
        # Graphical parameters
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=5, width_adjust=1.2, barwidth=.5,  # for the single plot
        scale="linear",
        style=True, color=None, grid=True, axescolor=None, titlecolor=None,
        suptitlecolor=None, suptitlesize=1.,  # multiplier of 15
        horizontal=None,
        labelrotation=75.,
        #
        print_info=True, res=False,
        *args, **kwargs):
    """
    Remarks
    -------
    - May be used also for numerics (but be careful when they have a lot of different values).
    - `most_common` applied before `sort_levels` -- good!

    Parameters
    ----------
    - most_common : 13; None or int
        if None all bars for all factor levels will be plotted;
        hence using None is dangerous if not sure how many levels there are;
        it's better to set big integer but no bigger then 100;
        otherwise plot may not be rendered at all if there are thousands of levels;

    Graphical parameters
    --------------------
    Currently there is only one plot in a figure for factors.
    It means that fig-size params are spurious but they are kept for
    consistency with plot_numeric() and for future development
    (other then bars plots for factors).

    #  Sizes for the whole figure
        These params overwrite single-plot-sizes params.
    figsize : None; tuple of numerics (figwidth, figheight)
    figwidth : None; numeric
    figheight : None; numeric

    #  Sizes for the single plot
        If width and height are None they are
    width : None; numeric
        = size if is None
    height : None; numeric
        = size * width_adjust if is None
    size : 5; numeric
        may be None only if width and height are not None or fig-sizes params are not None
    width_adjust : 1.2; numeric
        if width not set up directly then `width = size * width_adjust`
    barwidth : .5; numeric
        width of the single bar;
        if not None then width of the final plot is dependent on the number of levels
        and equals to `barwidth * nr_of_levels`;

    style : True; bool or str
        if True takes all the graphic parameters set externally (uses style from environment);
        if False is set to "dark_background";
        str must be a name of one of available styles: `plt.style.available`
    color : None; str
        color of lines and points for edge of bars;
        if None then set to "yellow" for style "black_background", else to "black";
    grid : False; bool or dict;
        if False then no grid is plotted (regrdless of style);
        if True then grid is plotted as ascribed to given style;
        in case of "black_background" it is dict(color='gray', alpha=.3)
        if dict then values from this dict will be applied as described in
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.grid.html
    titlecolor : None; str

    """
    # -----------------------------------------------------

    if isinstance(variable, str):
        varname = variable
        variable = data[variable]
    else:
        if varname is None:
            varname = coalesce(variable.name, "X")

    # -----------------------------------------------------
    #  info on raw variable
    var_info = cdf.info(pd.DataFrame(variable), what=["dtype", "oks", "oks_ratio", "nans_ratio", "nans", "uniques"])

    # -------------------------------------------------------------------------
    #  preparing data
    ascending = sort_levels if ascending is None else ascending

    variable_vc = variable.value_counts(ascending=ascending, dropna=dropna)
    n_levels = len(variable_vc)

    if most_common and most_common < n_levels:
        if title is None:
            title = f"{varname} \n most common {most_common} of {n_levels} values"  # ! 2 lines !
        levels_info_header = f" {varname} (most common {most_common} levels)"
        variable_vc = variable_vc.iloc[-most_common:] if ascending else variable_vc.iloc[:most_common]
    else:
        most_common = n_levels
        if title is None:
            title = varname
        levels_info_header = f" {varname} (all {n_levels} levels)"

    if sort_levels:
        try:
            variable_vc = variable_vc.sort_index(key=lambda k: float(k), ascending=ascending)
        except Exception:
            variable_vc = variable_vc.sort_index(ascending=ascending)

    if title_suffix:
        title = title + title_suffix

    # -----------------------------------------------------
    #  necessary for numerics turned to factors:
    levels = variable_vc.index.to_series().astype('str').values
    counts = variable_vc.values.tolist()

    var_variation = cdf.info(
        pd.DataFrame(variable),
        what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    if print_info:
        print(" 1. info on raw variable")
        print(var_info)
        print()
        print(" 2. statistics for processed variable (only most common values)")
        print(var_variation)
        print()

    if print_levels:
        # printing all levels is "dangerous" (may be a lot of them) and it's out of this function scope
        print(levels_info_header)
        print(variable_vc)

    # ----------------------------------------------------
    #  !!! result !!!

    result = {
        "title": title,
        "variable": variable,
        "info": var_info,
        "variation": var_variation,
        "distribution": variable_vc}  # variable after all prunings and transformations

    # ---------------------------------------------------------------------------------------------
    #  plotting

    # -------------------------------------------------------------------------
    #  style affairs

    style, color, grid, axescolor, suptitlecolor, titlecolor, _brightness, alpha = \
        h.style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, None, None, len(variable))

    # -------------------------------------------------------------------------
    #  sizes
    if figsize is None:

        n = min(most_common, n_levels)

        if figheight is None:
            height = size if height is None else height
            figheight = height

        if figwidth is None:
            if barwidth:
                width = barwidth * n if width is None else width
            else:
                width = size * width_adjust if width is None else width
            figwidth = width

    if horizontal is None:
        horizontal = len(levels) < 10

    if horizontal:
        figsize = figheight, figwidth + .8 * n / (n + 2)
        #
        levels = levels[::-1]
        counts = counts[::-1]
    else:
        figsize = figwidth, figheight

    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------------------------------------------------------
    #  plot

    if horizontal:
        bars = ax.barh(levels, counts, edgecolor=color, color='darkgray')
        #  ----------------------------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="y", grid=grid)
        # h.set_axescolor(ax, axescolor)
    else:
        bars = ax.bar(levels, counts, edgecolor=color, color='darkgray')
        #  ----------------------------
        h.set_yscale(ax, scale)
        h.set_grid(ax, off="x", grid=grid)
        # h.set_axescolor(ax, axescolor)
        ax.tick_params(axis='x', labelrotation=labelrotation)

    result['plot'] = dict(ax=ax, bars=bars)

    # -----------------------------------------------------
    #  final

    h.set_figtitle(fig, title, suptitlecolor, suptitlesize)

    fig.tight_layout()
    # plt.show()

    result['plot']["fig"] = fig

    return None if not res else result


# %%
plot_num = plot_numeric
plot_cat = plot_factor


# %%
def plot_variable(
        variable, data=None, varname=None,
        as_factor=None,  # !!!
        factor_threshold=13,
        # datetime=False,
        # Size parameters for numerics
        num_figsize=None, num_figwidth=None, num_figheight=None,    # for the whole figure
        num_width=None, num_height=None, num_size=4, num_width_adjust=1.2,
        # Size parameters for factors
        fac_figsize=None, fac_figwidth=None, fac_figheight=None,    # for the whole figure
        fac_width=None, fac_height=None, fac_size=5, fac_width_adjust=1.2, fac_barwidth=.5,
        # common (works if respective param for num/fac is None)
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=None, width_adjust=None, barwidth=None,  # for the single plot
        # title=None,
        # #
        # # factor params
        # most_common=13, sort_levels=False, print_levels=False, barwidth=.13,
        # #
        # # numeric params
        # what=[['hist', 'cloud'], ['boxplot', 'density'], ['agg', 'distr']],
        # # Variable modifications (before plotting)
        # upper=None, lower=None, exclude=None,
        # transform=False, agg=sum, bins=7,
        # n_obs=int(1e4), random_state=None, shuffle=False,
        # # Graphical parameters
        # lines=True, figsize=None, plotsize=4, width_adjust=1.2,
        # cmap="Paired",  # for coloring of bars in "hist" and respective points of "agg"
        # alpha=None, s=.2,   # alpha and size of a data point in a "cloud"
        # #
        # # common
        # style=True, color=None, grid=False, titlecolor=None,
        *args, **kwargs):
    """
    as_factor : None; bool
    factor_threshold : 13; int
        if numeric variable has less then `factor_threshold` then it will be
        treated as factor;
    """
    # -----------------------------------------------------

    if isinstance(variable, str):
        varname = variable
        variable = data[variable]
    else:
        if varname is None:
            varname = variable.name

    # -----------------------------------------------------

    if as_factor is None:
        as_factor = variable.dtype in ["category", "object", "str"] + ["datetime64[ns]", "datetime64"]
        if not as_factor:
            as_factor = variable.unique().shape[0] < factor_threshold

    # -----------------------------------------------------

    if as_factor:
        result = plot_factor(
            variable, data=data, varname=varname,
            figsize=coalesce(figsize, fac_figsize),
            figwidth=coalesce(figwidth, fac_figwidth),
            figheight=coalesce(figheight, fac_figheight),    # for the whole figure
            width=coalesce(width, fac_width),
            height=coalesce(height, fac_height),
            size=coalesce(size, fac_size),
            width_adjust=coalesce(size, fac_width_adjust),
            barwidth=coalesce(barwidth, fac_barwidth),
            *args, **kwargs)
    else:
        result = plot_numeric(
            variable, data=data, varname=varname,
            figsize=coalesce(figsize, num_figsize),
            figwidth=coalesce(figwidth, num_figwidth),
            figheight=coalesce(figheight, num_figheight),    # for the whole figure
            width=coalesce(width, num_width),
            height=coalesce(height, num_height),
            size=coalesce(size, num_size),
            width_adjust=coalesce(width_adjust, num_width_adjust),
            *args, **kwargs)

    return result

# %%
