#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Diagnostic plots for one variable
version: 1.0
type: module
keywords: [plot, variable, numeric, factor]
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
    date: 2021-10-30
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
from utils.builtin import coalesce
from utils.plots.plot_numeric import plot_numeric
from utils.plots.plot_factor import plot_factor


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
    *args, **kwargs
):
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
