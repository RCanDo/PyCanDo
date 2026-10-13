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
    - Here is dispatching method, i.e.
      it's only calling proper function with regard to type of a variable:
      for numeric time series it calls plot_ts_numeric() while
      for categorical (factor) ts it calls plot_ts_factor().
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

from .plot_ts_numeric import plot_ts_numeric
from .plot_ts_factor import plot_ts_factor
from utils.builtin import coalesce


# %%
def plot_ts(
        variable, time=None, data=None, varname=None, timename=None,
        as_factor=None,  # !!!
        factor_threshold=13,
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
        as_factor = variable.dtype in ["category", "object", "str"]
        if not as_factor:
            as_factor = variable.unique().shape[0] < factor_threshold

    # -----------------------------------------------------

    if as_factor:
        result = plot_ts_factor(
            variable, time=time, data=data,
            varname=varname, timename=timename,
            *args, **kwargs)
    else:
        result = plot_ts_numeric(
            variable, time=time, data=data,
            varname=varname, timename=timename,
            *args, **kwargs)

    return result


# %%
def lines_at_moments(lines_at, line_color='r', line_alpha=.3, plotter=plot_ts, **kwargs):
    """
    Adding vertical lines at given moments (`lines_at`)
    to the plot created by `plotter`.
    The plotter should
    - create plot with x-axis in datetime format,
    - return dictionary `res` where `res['plot']['ax']` is a matplotlib axis object;
    currently only plot_datetime() or plot_ts() and its subroutines
    plot_ts_numeric(), plot_ts_factor() (all in this module).
    """
    lower = coalesce(kwargs.get("lower", None), kwargs.get("lower_t", None))
    upper = coalesce(kwargs.get("upper", None), kwargs.get("upper_t", None))
    if lower:
        lines_at = lines_at[lines_at >= pd.to_datetime(lower)]
    if upper:
        lines_at = lines_at[lines_at <= pd.to_datetime(upper)]

    res = plotter(res=True, **kwargs)
    ymin, ymax = res['plot']['ax'].get_ylim()
    # xmin, xmax = res['plot']['ax'].get_xlim()
    res['plot']['ax'].vlines(lines_at, ymin=ymin, ymax=ymax, color=line_color, alpha=line_alpha)


# %%
