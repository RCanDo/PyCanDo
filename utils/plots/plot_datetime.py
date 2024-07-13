#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Plotting datetime
version: 1.0
type: module
keywords: [plot, datetime]
description: |
    PLotting datatime variable:
    x - dates
    y - frequency of each date
    Interface similar to plot_variable()
    however only one type of plot is available now, i.e. frequency plot.
    One may transform datetime in a similar way as
    variable in plot_variable().
content:
remarks:
todo:
    - "reversed" plot, i.e. dates which do not appear as they should
      if data assumed to be continuous with given (or inferred) frequency;
      for discovering lacks of (holes in) data
sources:
file:
    date: 2022-08-31
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
# %% imports
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt

# from utils.builtin import coalesce

import utils.df as cdf
import utils.plots.helpers as h


# %%
def plot_datetime(
        variable, data=None, varname=None, title=None, title_suffix=None,
        # Variable modifications (before plotting)
        lower=None, upper=None, exclude=None,                                   # ???
        print_most_common=13,  # None,
        transform=False,
        upper_t=None, lower_t=None, exclude_t=None,
        #
        # Graphical parameters
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=5, width_adjust=2.5,  # for the single plot
        scale="linear",
        # lines=True,
        # cmap="ak01",  # for coloring of bars in "hist" and respective points of "agg"
        alpha=1, s=9, brightness=None,  # alpha, size and brightness of a data point
        style=True, color=None, grid=True, axescolor=None, titlecolor=None,
        suptitlecolor=None, suptitlesize=1.,  # multiplier of 15
        # horizontal=None,
        labelrotation=0.,   # in degrees, 90. is perpendicular
        #
        # print_levels=False,
        print_info=True, res=False,
        *args, **kwargs):
    """
    PLotting datatime variable:
    x - dates;
    y - frequency of each date.
    Interface similar to plot_variable()
    however only one type of plot is available now, i.e. frequency plot.
    One may transform datetime in a similar way as
    variable in plot_variable().
    """

    # -------------------------------------------------------------------------
    #  loading data

    variable, varname = h.get_var_and_name(variable, data, varname, "date-time")

    # -----------------------------------------------------
    #  info on raw variable
    var_info = cdf.info(pd.DataFrame(variable), what=["dtype", "oks", "oks_ratio", "nans_ratio", "nans", "uniques"])

    if print_info:
        print(" 1. info on raw variable")
        print(var_info)

    # -------------------------------------------------------------------------
    #  preparing data

    variable = variable.dropna()                                                # !!!
    variable = pd.to_datetime(variable)                                         # !!!

    # -----------------------------------------------------
    #  transformation and clipping

    lower, upper, exclude, lower_t, upper_t, exclude_t = \
        [h.to_datetime(t) for t in [lower, upper, exclude, lower_t, upper_t, exclude_t]]

    variable, transname = h.clip_transform(
        variable,
        lower, upper, exclude,
        transform, lower_t, upper_t, exclude_t, "T")

    variable_vc = variable.value_counts()

    # -----------------------------------------------------
    #  statistics for processed variable

    var_variation = cdf.summary(
        pd.DataFrame(variable),
        what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    if print_info:
        print()
        print(" 2. statistics for processed variable (excluded empty records and within limits)")
        print(var_variation)
    if print_most_common:   # int, None
        print()
        print("  most common:")
        print(variable_vc[:print_most_common])

    # -----------------------------------------------------
    #  title

    if not title:
        lower = h.datetime_to_str(lower)
        upper = h.datetime_to_str(upper)
        title = h.make_title(varname, lower, upper, transname, lower_t, upper_t)

    if title_suffix:
        title = title + title_suffix

    #  ----------------------------------------------------
    #  !!! result !!!

    result = {"title": title,
              "variable": variable,
              "info": var_info,
              "variation": var_variation,
              "distribution": variable_vc}  # variable after all pruning and transformations

    # ---------------------------------------------------------------------------------------------
    #  plotting

    # -------------------------------------------------------------------------
    #  style affairs

    style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha = \
        h.style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha, len(variable))

    # -------------------------------------------------------------------------
    #  plot types

    def frequencies(ax, title):
        """"""
        # set_title(ax, title, titlecolor)
        #  ---------
        res = ax.scatter(variable_vc.index, variable_vc.values, s=s, color=color, alpha=alpha, **kwargs)
        #  ---------
        h.set_yscale(ax, scale)
        h.set_grid(ax, grid=grid)
        # h.set_axescolor(ax, axescolor)
        ax.set_ylabel("nr of events")
        ax.tick_params(axis='x', labelrotation=labelrotation)    # !!! not always good; to be optional
        return ax, res

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
        nonlocal variable_vc

        vc_max = variable_vc.max() if len(variable_vc) > 0 else 1

        if figsize is None:

            height_adjust = min(vc_max / 10, 1)

            if figheight is None:
                height = max(size * height_adjust, 2) if height is None else height
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

    ax, freqs = frequencies(ax, title)

    result['plot'] = dict(ax=ax, freqs=freqs)

    # -----------------------------------------------------
    #  final

    h.set_figtitle(fig, title, suptitlecolor, suptitlesize)

    fig.tight_layout()
    # plt.show()

    result['plot']["fig"] = fig

    return None if not res else result


# %%
