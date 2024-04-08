#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Diagnostic plots for one variable
version: 1.0
type: module
keywords: [plot, factor, categorical, barplot]
description: |
    Custom diagnostic plots for one categorical (factor) variable:
        - barplot.
    Maximum flexibility (lots of parameters) but with sensible defaults.
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

import pandas as pd

# import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.builtin import coalesce
import utils.df as cdf
import utils.plots.helpers as h


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


# %% alias
plot_cat = plot_factor
