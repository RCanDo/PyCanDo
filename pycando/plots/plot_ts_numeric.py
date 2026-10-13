#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Plotting time series
version: 1.0
type: module
keywords: [plot, datetime, time series]
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
    - currently only numeric values tackled
    - Ideally, this type of plot should be part of plot_covariates(),
      however this is technically challenging as there is
      large number of minor differences.
todo:
    - multivariate time series
    - factor valued ts
    - density and orientation of time tags
    - merging with plot_variable() ???
sources:
file:
    date: 2022-08-31
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import pandas as pd

import matplotlib.pyplot as plt

from utils.builtin import coalesce
import utils.df as cdf
import utils.plots.helpers as h


# %%
def plot_ts_numeric(
        variable, time=None, data=None,
        varname=None, timename=None, title=None, title_suffix=None,
        ignore_index=False,
        # Variables modifications (before plotting)
        lower_v=None, upper_v=None, exclude_v=None,
        lower_t=None, upper_t=None, exclude_t=None,
        transform_v=None, transform_t=None,
        lower_t_v=None, upper_t_v=None, exclude_t_v=None,
        lower_t_t=None, upper_t_t=None, exclude_t_t=None,
        n_obs=None, random_state=None,  # shuffle=False,                        #???
        # Graphical parameters
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=5, width_adjust=2.5,  # for the single plot
        scale="linear",
        # lines=True,
        # cmap="ak01",  # for coloring of bars in "hist" and respective points of "agg"
        alpha=None, s=1, brightness=None,  # alpha, size and brightness of a data point
        style=True, color=None, grid=True, axescolor=None, titlecolor=None,
        suptitlecolor=None, suptitlesize=1.,  # multiplier of 15
        # horizontal=None,
        #
        # print_levels=False,
        tex=False,  # varname & covarname passed in TeX format e.g. "\\hat{Y}" (double`\` needed)
        print_info=True, res=False,
        *args, **kwargs):
    """
    Plotting time series given as
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

    # !!! get  color, s, alhpa  from data if they are proper column names and not color names !!!

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

    del df0

    # -------------------------------------------------------------------------
    #  preparing data

    variable, _ = h.clip_transform(variable, lower_v, upper_v, exclude_v)

    lower_t, upper_t, exclude_t = [h.to_datetime(t) for t in [lower_t, upper_t, exclude_t]]

    time, _ = h.clip_transform(time, lower_t, upper_t, exclude_t)

    # aligning data
    df0 = pd.concat([variable, time], axis=1)
    df0.columns = [varname, timename]
    df0.dropna(inplace=True)
    # df0.index = range(df0.shape[0])
    # # or df = df.reset_index() # but it creates new column with old index -- mess

    variable = df0[varname]
    time = df0[timename]

    df = df0
    # df0 -- data not transformed (however clipped and .dropna())
    # df -- data potentially transformed (or just copy of df0 if no tranformations)

    # -----------------------------------------------------
    #  transforms

    variable, transname_v = h.clip_transform(
        variable, None, None, None,
        transform_v, lower_t_v, upper_t_v, exclude_t_v, "T_v")

    lower_t_t = pd.to_datetime(lower_t_t) if lower_t_t else None
    upper_t_t = pd.to_datetime(upper_t_t) if upper_t_t else None
    exclude_t = pd.to_datetime(exclude_t_t) if exclude_t_t else None

    lower_t_t, upper_t_t, exclude_t_t = [h.to_datetime(t) for t in [lower_t_t, upper_t_t, exclude_t_t]]

    time, transname_t = h.clip_transform(
        time, None, None, None,
        transform_t, lower_t_t, upper_t_t, exclude_t_t, "T_t")

    # aligning data
    # !!! make it robust on (*): when False passed data_were_processed=True -- bad!
    transforms = [transform_v, lower_t_v, upper_t_v, exclude_t_v, transform_t, lower_t_t, upper_t_t, exclude_t_t]
    if any(transforms):
        df = pd.concat([variable, time], axis=1)
        df.columns = [varname, time]
        df.dropna(inplace=True)
        # df.index = range(df.shape[0])

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

        title = h.make_title(varname, lower_v, upper_v, transname_v, lower_t_v, upper_t_v, tex) + \
            " ~ " + \
            h.make_title(timename, lower_t, upper_t, transname_t, lower_t_t, upper_t_t, tex)

    if title_suffix:
        title = title + title_suffix

    #  ----------------------------------------------------
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

    if isinstance(alpha, str):
        alpha = data[alpha]

    if isinstance(s, str):
        s = data[s]

    if isinstance(color, str) and not h.is_mpl_color(color) and color in data.columns:
        color = data[color]

    style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha = \
        h.style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha, N)

    # -----------------------------------------------------
    # sampling and aligning color, size, alpha (if they are series)

    df = h.sample(df, n_obs, False, random_state)   # don't shuffle here! it's time series !!!
    df, color, s, alpha = cdf.align_nonas(df, color=color, s=s, alpha=alpha)
    time = df[timename]
    variable = df[varname]

    # -------------------------------------------------------------------------
    #  plot types

    def scatter(ax, title):
        """"""
        # set_title(ax, title, titlecolor)
        # ---------
        scat = ax.scatter(time, variable, s=s, color=color, alpha=alpha, **kwargs)
        # ---------
        h.set_yscale(ax, scale)
        h.set_grid(ax, grid=grid)
        # set_axescolor(ax, axescolor)
        # ax.tick_params(axis='x', labelrotation=75.)    # !!! not always good; to be optional
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

        if figsize is None:

            if figheight is None:
                height = size if height is None else height
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
