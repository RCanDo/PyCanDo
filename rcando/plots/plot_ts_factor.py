#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Plotting categorical time series
version: 1.0
type: module
keywords: [plot, datetime, categorical time series]
description: |
    Plotting time series given as
    1. values pd.Series against time valued pd.Series, or
    2. pd.Series with time index.
    Interface similar to plot_covariates()
    however only one type of plot is available now.
    One may transform values and time in a similar way as
    variable and covariate in plot_covariates().
content:
    -
remarks:
    - Ideally, this type of plot should be part of plot_covariates(),
      however this is technically challenging as there is
      large number of minor differences.
todo:
    - density and orientation of time tags
    - merging with plot_variable() ???
sources:
file:
    usage:
        interactive: False
        terminal: False
    date: 2022-09-13
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from common.builtin import coalesce
import common.df as cdf
import common.plots.helpers as h


# %%
def plot_ts_factor(
        variable, time=None, data=None,
        varname=None, timename=None, title=None, title_suffix=None,
        ignore_index=False,
        # Variables modifications (before plotting)
        most_common=13, sort_levels=False, print_levels=False,
        lower_t=None, upper_t=None, exclude_t=None,
        transform_t=None,
        lower_t_t=None, upper_t_t=None, exclude_t_t=None,
        n_obs=None, random_state=None,  # shuffle=False,                        #???
        # Graphical parameters
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=5, width_adjust=2.5,  # for the single plot
        scale="linear",
        # lines=True,
        cmap="ak01",  # for coloring of bars in "hist" and respective points of "agg"
        alpha=.5, s=10, brightness=None,  # alpha, size and brightness of a data point
        style=True, color=True, grid=True, axescolor=None, titlecolor=None,
        suptitlecolor=None, suptitlesize=1.,  # multiplier of 15
        # horizontal=None,
        labelrotation=0.,   # in degrees, 90. is perpendicular
        #
        # print_levels=False,
        tex=False,  # varname & covarname passed in TeX format e.g. "\\hat{Y}" (double`\` needed)
        print_info=True, res=False,
        *args, **kwargs):
    """
    Plotting categorical time series given as
    1. values pd.Series against time valued pd.Series, or
    2. pd.Series with time index.
    Interface similar to plot_covariates()
    however only one type of plot is available now.
    One may transform values and time in a similar way as
    variable and covariate in plot_covariates().
    """

    # -------------------------------------------------------------------------
    #  loading data

    variable, varname = h.get_var_and_name(variable, data, varname, "variable")

    time = coalesce(time, variable.index)       # for TS as pd.Series with datetime index
    time, timename = h.get_var_and_name(time, data, timename, "time")
    time = pd.to_datetime(time)

    # !!! index is ABSOLUTELY CRITICAL here !!!
    if ignore_index:
        variable, time, color, s, alpha = cdf.align_indices(variable, time, color, s, alpha)

    # -----------------------------------------------------
    #  info on raw data

    df0 = pd.concat([variable, time], axis=1)
    df0.columns = [varname, timename]

    df0_info = cdf.info(df0, what=["dtype", "oks", "oks_ratio", "nans_ratio", "nans", "uniques"])
    if print_info:
        print(" 1. info on raw data")
        print(df0_info)

    variable = df0[varname]
    time = df0[timename]

    # del df0

    # -------------------------------------------------------------------------
    #  preparing data

    # -----------------------------------------------------
    #  time clipping/transforming

    lower_t, upper_t, exclude_t = [h.to_datetime(t) for t in [lower_t, upper_t, exclude_t]]

    time, _ = h.clip_transform(time, lower_t, upper_t, exclude_t)

    # ---------------------------------
    #  transforms

    lower_t_t = pd.to_datetime(lower_t_t) if lower_t_t else None
    upper_t_t = pd.to_datetime(upper_t_t) if upper_t_t else None
    exclude_t = pd.to_datetime(exclude_t_t) if exclude_t_t else None

    lower_t_t, upper_t_t, exclude_t_t = [h.to_datetime(t) for t in [lower_t_t, upper_t_t, exclude_t_t]]

    time, transname_t = h.clip_transform(
        time, None, None, None,
        transform_t, lower_t_t, upper_t_t, exclude_t_t, "T_t")

    # ---------------------------------
    # aligning

    df = pd.concat([variable, time], axis=1)
    df.columns = [varname, timename]
    df.dropna(inplace=True)
    # df.index = range(df.shape[0])
    # # or df = df.reset_index() # but it creates new column with old index -- mess

    variable = df[varname]
    time = df[timename]

    # -----------------------------------------------------
    #  variable's most_common (after time clipping/transforming)

    variable_vc = variable.value_counts()
    n_levels = len(variable_vc)

    if print_levels:
        # prints ALL levels regardless of `most_common` BUT AFTER time processing !!!
        if sort_levels:
            try:
                print(variable_vc.sort_index(key=lambda k: float(k)))
            except Exception:
                print(variable_vc.sort_index())
        else:
            print(variable_vc)

    if most_common and most_common < n_levels:
        if title is None:
            title_var = f"{varname} [most common {most_common} of {n_levels} values] \n"
            # ! 2 lines ! but second empty -- ready for  time-title
        variable_vc = variable_vc.iloc[:most_common]

        levels = variable_vc.index.tolist()
        variable = variable[variable.isin(levels)]

    else:
        most_common = n_levels
        if title is None:
            title_var = varname

        levels = variable_vc.index.tolist()

    # # !!!  move it to proper place !!!
    # if sort_levels:
    #     try:
    #         variable_vc = variable_vc.sort_index(key=lambda k: float(k))
    #     except Exception:
    #         variable_vc = variable_vc.sort_index()

    # ---------------------------------
    # aligning

    df = pd.concat([variable, time], axis=1)
    df.columns = [varname, timename]
    df.dropna(inplace=True)
    # df.index = range(df.shape[0])
    # # or df = df.reset_index() # but it creates new column with old index -- mess

    variable = df[varname]
    time = df[timename]

    # -----------------------------------------------------
    #  statistics for processed data
    df_variation = cdf.summary(
        df, what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    df_distribution = cdf.summary(
        df, what=["range", "iqr", "mean", "median", "min", "max", "negatives", "zeros", "positives"])

    if print_info:  # of course!
        print()
        print(" 2. statistics for processed data")
        print(df_variation)
        print()
        print(df_distribution)

    # -----------------------------------------------------
    #  title

    if title is None:

        lower_t = h.datetime_to_str(lower_t)
        upper_t = h.datetime_to_str(upper_t)
        lower_t_t = h.datetime_to_str(lower_t_t)
        upper_t_t = h.datetime_to_str(upper_t_t)

        title = title_var + \
            " ~ " + \
            h.make_title(timename, lower_t, upper_t, transname_t, lower_t_t, upper_t_t, tex)

    if title_suffix:
        title = title + title_suffix

    # ----------------------------------------------------
    #  !!! result !!!

    result = {
        "title": title,
        "df0": df0,  # unprocessed
        "df": df,    # processed
        "info": df0_info,
        "variation": df_variation,
        "distribution": df_distribution, }  # variable after all prunings and transformations

    # ---------------------------------------------------------------------------------------------
    #  plotting

    # -------------------------------------------------------------------------
    #  style affairs

    N = len(variable) if not n_obs else min(len(variable), int(n_obs))

    style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha = \
        h.style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha, N)

    df = h.sample(df, n_obs, False, random_state)
    time = df[timename]
    variable = df[varname]

    # -------------------------------------------------------------------------
    #  plot types

    # def cats_and_colors(factor, most_common):
    #     """helper function
    #     most_common : None; pd.Series
    #         table of most common value counts = variable.value_counts[:most_common]
    #         got from  to_factor(..., most_common: int)
    #     Returns
    #     -------
    #     cats : categories (monst common) of a factor
    #     cat_colors : list of colors of length len(cats)
    #     cmap : listed color map as defined in matplotlib
    #     """
    #     cats = factor.cat.categories.to_list()
    #     if most_common is not None:
    #         cats = [cat for cat in cats if cat in most_common]
    #     #
    #     cat_colors = plt.colormaps['hsv'](np.linspace(0.1, 0.9, len(cats)))
    #     cmap = ListedColormap(cat_colors)
    #     return cats, cat_colors, cmap

    def scatter(ax, title):
        """"""
        nonlocal levels  # ~= cats : list
        nonlocal color
        nonlocal cmap
        nonlocal brightness
        # set_title(ax, title, titlecolor)
        # ---------

        dfp = pd.pivot_table(df, index=timename, columns=varname, aggfunc=len)
        # assert levels == dfp.columns.tolist()

        cmap = h.mpl.cm.get_cmap(cmap)
        cat_colors = cmap(np.linspace(0, 1, len(levels)))[:, :3]  # no alpha
        cat_colors = h.brightness(cat_colors, brightness)

        scat = dict()
        levels = dict()     # rewrting

        for c, cat in enumerate(dfp.columns):
            levels[c] = cat
            var = dfp[cat]
            var.dropna(inplace=True)
            color = color if isinstance(color, str) else ([tuple(cat_colors[c, :3])] * len(var))
            #    # color is str or True -- see h.style_affairs()
            scat[cat] = ax.scatter(var.index, [c] * len(var), s=var * s, alpha=alpha, c=color, **kwargs)  # , cmap=cmap)

        # ---------
        # h.set_yscale(ax, scale)
        ax.set_yticks(list(levels.keys()), list(levels.values()))
        h.set_grid(ax, grid=grid)
        # set_axescolor(ax, axescolor)
        ax.tick_params(axis='x', labelrotation=labelrotation)
        return ax, scat

    # -------------------------------------------------------------------------
    #  plotting procedure

    # -----------------------------------------------------
    #  sizes

    def set_fig():

        nonlocal size
        nonlocal height
        nonlocal width
        nonlocal figsize
        nonlocal figheight
        nonlocal figwidth
        nonlocal levels

        if figsize is None:

            if figheight is None:
                height = max(size * len(levels) / 10, 2) if height is None else height
                figheight = height

            if figwidth is None:
                width = size * width_adjust if width is None else width
                figwidth = width

        figsize = figwidth, figheight

        fig, ax = plt.subplots(figsize=figsize)

        return fig, ax

    # -----------------------------------------------------
    #  core

    fig, ax = set_fig()

    ax, scat = scatter(ax, title)

    result['plot'] = dict(ax=ax, scat=scat)

    # -------------------------------------------------------------------------
    #  final

    h.set_figtitle(fig, title, suptitlecolor, suptitlesize)

    fig.tight_layout()
    # plt.show()

    result['plot']["fig"] = fig

    return None if not res else result

# %%
