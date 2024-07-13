#! python3
# -*- coding: utf-8 -*-
"""
---
title: Variable (y) vs covariate (x)
version: 1.0
type: module
keywords: [plot, ]
description: |
    What is the dependence of a variable (y) on the covariate (x)?
    There are 4 cases:
        (y is numeric / categorical) x (x is numeric / categorial)
    where categorical ~ numeric is turned to numeric ~ categorical
    (as it's more straitforward to show dependency this way - see todo however);
     in this case there are 3 main types of plots:
         - grouped cloud
         - densities (separate density of y for each level of x)
         - boxplots (or each level of x)
    numeric ~ numeric is just an ordinary scatter-plot
    while cat ~ cat is a bar-chart.
    As for the plot_covariate() the idea was to make it fully automated
    but also very flexible - lots of parameters but with sensible defaults.
content:
    -
remarks:
todo:
    - categorical ~ numeric  may be shown via multinomial logistic model,
      i.e. probability of each category wrt to value of x;
    - violinplots  for  numeric ~ factor (~432);
    - factor ~ factor (~695)  is in very crude version;
    - barchart (~475): legend (bbox_to_anchor, loc, ...);
      figure size for large amount of levels and long level names;
    -
sources:
file:
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
import statsmodels.api as sm
from scipy.stats import gaussian_kde

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # LinearSegmentedColormap

import utils.df as udf
from utils.builtin import coalesce
import utils.plots.helpers as h
from utils.plots.rocs import rocs as rocs0


# %% plot_()
def plot_covariates(
        variable, covariate, data=None,
        varname=None, covarname=None, title=None, title_suffix=None,
        ignore_index=False,
        factor_threshold=13,
        as_factor_y=None, as_factor_x=None,
        what=["grouped_cloud", "densities", "boxplots"],
        # Variables modifications (before plotting)
        lower_y=None, upper_y=None, exclude_y=None,
        lower_x=None, upper_x=None, exclude_x=None,
        transform_y=None, transform_x=None,
        lower_t_y=None, upper_t_y=None, exclude_t_y=None,
        lower_t_x=None, upper_t_x=None, exclude_t_x=None,
        #
        bins=21, smooth=.5,  # 0 or 1 for OLS (check it!)
        qq_plot=False,       # add qq-plot for  num_vs_num  plot
        sort_levels=True, legend=True, axes_labels=True, print_levels=True, bar_height=.5,
        n_obs=int(1e4), random_state=None, shuffle=False,        # ???
        most_common=13,
        # Graphical parameters
        figsize=None, figwidth=None, figheight=None,  # for the whole figure
        width=None, height=None, size=5, width_adjust=1.2,  # for the single plot
        scale="linear", xscale=None, yscale=None,
        lines=True,  # not used here yet!
        cmap="ak01",  # for coloring of bars in "hist" and respective points of "agg"
        color=None, s=9, alpha=None, brightness=None,  # alpha, size and brightness of a data point
        style=True, grid=True, axescolor=None, titlecolor=None,
        suptitlecolor=None, suptitlesize=1.,  # multiplier of 15
        #
        tex=False,  # varname & covarname passed in TeX format e.g. "\\hat{Y}" (double`\` needed)
        print_info=True, res=False,
        *args, **kwargs):
    """
    Remarks
    -------
    1. It is assumed that `variable` may be passed as pd.Series
    while `covariate` as a string indicating column of `data`.
    `variable` and `covariate` may have different indices while they may be irrelevant
    (to be ignored).
    Thus `ignore_index` is provided but it has different meaning from the same parameter of pd.concat():
    here it means that if `ignore_index=True` then we ignore indices of `variable` and `covariate`
    and make one index common to both of them based solely on the elements order
    (thus number of elements must be the same in both series);
    It is critical for proper aligning of both data series.
    Default value for `ignore_index` is False what means that we pay attention to
    both indices and align two series according to indices values (like in pd.concat()).

    2. Plots are drawn as `variable ~ covariate` i.e.
    `variable` serves as `y` (_dependent_ or _explained_ variable)
    and `covariate` serves as `x` (_independent_ or _explanatory_ variable).
    All parameter names where 'x' or 'y' is used are based on this convention.
    It was preferred to use 'x' or 'y' for its brevity;
    However, `variable` and `covariate` are used for the first two parameters
    (instead of 'y' and 'x') to convey their meaning and objective of the whole function:
    explain (via plots) `variable` (`y`) with `covariate` (`x`).

    3. `most_common` applied before `sort_levels`.
    """
    # -------------------------------------------------------------------------
    #  loading data

    variable, varname = h.get_var_and_name(variable, data, varname, "Y")
    covariate, covarname = h.get_var_and_name(covariate, data, covarname, "X")
    if varname == covarname:
        covarname += "_0"

    # !!! index is ABSOLUTELY CRITICAL here !!!
    if ignore_index:
        variable, covariate, color, s, alpha = udf.align_indices(variable, covariate, color, s, alpha)

    # -----------------------------------------------------
    #  info on raw data

    df0 = pd.concat([variable, covariate], axis=1)
    df0.columns = [varname, covarname]

    df0_info = udf.info(df0, what=["dtype", "oks", "oks_ratio", "nans_ratio", "nans", "uniques"])
    if print_info:
        print(" 1. info on raw data")
        print(df0_info)

    variable = df0[varname]
    covariate = df0[covarname]

    del df0

    # -------------------------------------------------------------------------
    #  preparing data

    # this makes sense also for factors although not always
    variable, _ = h.clip_transform(variable, lower_y, upper_y, exclude_y)
    covariate, _ = h.clip_transform(covariate, lower_x, upper_x, exclude_x)

    # it's here because we may turn numeric to factor after clipping
    is_factor_y, variable, variable_vc = h.to_factor(
        variable, as_factor_y, most_common=most_common, factor_threshold=factor_threshold)
    is_factor_x, covariate, covariate_vc = h.to_factor(
        covariate, as_factor_x, most_common=most_common, factor_threshold=factor_threshold)

    # aligning data
    df0 = pd.concat([variable, covariate], axis=1)
    df0.columns = [varname, covarname]
    df0.dropna(inplace=True)

    variable = df0[varname]
    covariate = df0[covarname]

    df = df0
    # df0 -- data not transformed (however clipped and .dropna())
    # df  -- data potentially transformed (or just copy of df0 if no tranformations)

    # -----------------------------------------------------
    #  transforms

    transname_y = None
    if not is_factor_y:
        variable, transname_y = h.clip_transform(
            variable, None, None, None,
            transform_y, lower_t_y, upper_t_y, exclude_t_y, "T_y")

    transname_x = None
    if not is_factor_x:
        covariate, transname_x = h.clip_transform(
            covariate, None, None, None,
            transform_x, lower_t_x, upper_t_x, exclude_t_x, "T_x")

    # aligning data
    # !!! make it robust on (*): when False passed data_were_processed=True -- bad!
    transforms = [transform_y, lower_t_y, upper_t_y, exclude_t_y, transform_x, lower_t_x, upper_t_x, exclude_t_x]
    if any(transforms):
        df = pd.concat([variable, covariate], axis=1)
        df.columns = [varname, covarname]
        df.dropna(inplace=True)
        # df.index = range(df.shape[0])

        variable = df[varname]
        covariate = df[covarname]

        data_were_processed = True     # (*)
    else:
        data_were_processed = False    # (*)

    # print(data_were_processed)

    # -----------------------------------------------------
    #  statistics for processed data

    df_variation = udf.summary(
        df, what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    df_distribution = udf.summary(
        df, what=["range", "iqr", "mean", "median", "min", "max", "negatives", "zeros", "positives"])

    # -----------------------------------------------------
    #  title

    if title is None:

        title = h.make_title(varname, lower_y, upper_y, transname_y, lower_t_y, upper_t_y, tex) + \
            " ~ " + \
            h.make_title(covarname, lower_x, upper_x, transname_x, lower_t_x, upper_t_x, tex)
        # the same for  numeric ~ factor  and  factor ~ numeric

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
        "distribution": df_distribution,  # variable after all prunings and transformations
        "plot": dict()
    }

    # ---------------------------------------------------------------------------------------------
    #  plotting

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

    # -----------------------------------------------------
    # sampling and aligning color, size, alpha (if they are series)

    def sample_and_align(variable, covariate, n_obs=n_obs, shuffle=shuffle, random_state=random_state,
                         color=color, s=s, alpha=alpha, color_data=color_data):
        df = pd.concat([variable, covariate], axis=1)
        df, color, s, alpha, color_data = udf.align_sample(df, n_obs, shuffle, random_state,
                                                           color=color, s=s, alpha=alpha, color_data=color_data)
        df, color, s, alpha, color_data = udf.align_nonas(df, color=color, s=s, alpha=alpha, color_data=color_data)
        variable = df.iloc[:, 0]
        covariate = df.iloc[:, 1]
        return variable, covariate, color, s, alpha, color_data

    # -------------------------------------------------------------------------
    #  plot types

    # -----------------------------------------------------
    #  numeric ~ numeric

    def qq(covariate, variable):
        qqx = [covariate.quantile(q / 10) for q in range(11)]
        qqy = [variable.quantile(q / 10) for q in range(11)]
        return qqx, qqy

    def scatter_hist(ax, ax_histx, ax_histy):
        """helper for cloud()
        scatter plot of variable vs covariate and
        side histograms for each var (marginal distributions)
        """
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        scatter = ax.scatter(covariate, variable, s=s, color=color_data, alpha=alpha)
        if qq_plot:
            qqx, qqy = qq(covariate, variable)
            ax.plot(qqx, qqy, color=mpl.colors.to_rgba(color, .5), marker="*")
        else:
            qqx, qqy = None, None

        # now determine nice limits by hand:
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax/binwidth) + 1) * binwidth

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        histx = ax_histx.hist(covariate, bins=bins, color=mpl.colors.to_rgba(color, .6))
        histy = ax_histy.hist(variable, bins=bins, color=mpl.colors.to_rgba(color, .6), orientation='horizontal')

        result = dict(scatter=scatter, histx=histx, histy=histy, qqx=qqx, qqy=qqy)
        return dict(ax=ax, result=result)

    def smoother(ax):
        """helper for cloud()
        lowess trend of variable vs covariate
        """
        xx = np.linspace(min(covariate), max(covariate), 100)
        smoothed = sm.nonparametric.lowess(
            exog=covariate, endog=variable,
            xvals=xx,
            frac=smooth)

        ax.plot(xx, smoothed, c="r" if color != 'r' else 'k')

        result = dict(xx=xx, smoothed=smoothed)
        return dict(ax=ax, result=result)

    def cloud(fig, title="scatter"):
        """
        On how to get side histograms
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
        """
        # set_title(ax, title, titlecolor)
        # #  ---------

        # definitions for the axes
        left, width = 0.1, 0.7
        bottom, height = 0.1, 0.7
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.145]
        rect_histy = [left + width + spacing, bottom, 0.149, height]

        # start with a square Figure
        # fig = plt.figure(figsize=(8, 8))

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # use the previously defined function
        sh = scatter_hist(ax, ax_histx, ax_histy)
        sm = smoother(sh['ax'])
        ax = sm['ax']

        # plt.show()
        if axes_labels:
            ax.set_ylabel(varname)
            ax.set_xlabel(covarname)

        #  ---------
        if grid:
            if not isinstance(grid, bool):
                ax.grid(**grid)
                ax_histx.grid(**grid)
                ax_histy.grid(**grid)
        else:
            ax.grid(visible=False, axis="both")
            ax_histx.grid(visible=False, axis="both")
            ax_histy.grid(visible=False, axis="both")

        axes = dict(ax=ax, ax_histx=ax_histx, ax_histy=ax_histy)
        result = dict(scatter_hist=sh['result'], smoother=sm['result'])

        return dict(fig=fig, axes=axes, result=result)

    # -----------------------------------------------------
    #  numeric ~ factor

    def cats_and_colors(factor, most_common, cmap):
        """helper function
        most_common : None; pd.Series
            table of most common value counts = variable.value_counts[:most_common]
            got from  to_factor(..., most_common: int)
        Returns
        -------
        cats : categories (monst common) of a factor
        cat_colors : list of colors of length len(cats)
        cmap : listed color map as defined in matplotlib
        """
        cats = factor.cat.categories.to_list()
        if most_common is not None:
            cats = [cat for cat in cats if cat in most_common]
        #
        cat_colors = plt.colormaps[cmap](np.linspace(0.1, 0.9, len(cats)))
        cmap = ListedColormap(cat_colors)
        return cats, cat_colors, cmap

    def grouped_cloud(ax, variable, factor, cats, cat_colors, cmap, title="grouped cloud", sort=sort_levels):
        """
        """
        h.set_title(ax, title, titlecolor)
        #  ---------

        dff = pd.concat([variable, factor, factor.cat.codes], axis=1)
        dff.columns = [variable.name, factor.name, "cats"]
        if sort:
            dff.sort_values(by="cats", ignore_index=True, inplace=True)

        dff = dff[dff[factor.name].isin(cats)]     # sorting is retained !

        scatter = ax.scatter(dff[variable.name], dff.index, c=dff['cats'], cmap=cmap, s=s, alpha=alpha)

        if axes_labels:
            ax.set_xlabel(varname)
            # ax.set_ylabel('id')
        #  ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        #
        result = dict(scatter=scatter, cat_colors=cat_colors, dff=dff)
        return dict(ax=ax, result=result)

    def densities(ax, variable, factor, cats, cat_colors, cmap, title="densities by levels", legend=legend):
        """
        """
        h.set_title(ax, title, titlecolor)
        #  ---------
        result = dict()
        for cat, col in reversed(list(zip(cats, cat_colors))):
            result[cat] = dict()
            vv = variable[factor == cat]
            if len(vv) > 1:
                try:
                    result[cat]['kde'] = gaussian_kde(vv.astype(float))
                except Exception:
                    result[cat]['kde'] = gaussian_kde(vv)
                xx = np.linspace(min(vv), max(vv), 200)
                lines = ax.plot(xx, result[cat]['kde'](xx), color=col, label=cat)
                result[cat]['xx'] = xx
                result[cat]['lines'] = lines
        if legend:
            ax.legend(title=covarname)
        if axes_labels:
            ax.set_xlabel(varname)
        #  ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        #
        return dict(ax=ax, result=result)

    def distr(ax, variable, factor, cats, cat_colors, cmap, title="distributions by levels", legend=legend):
        """
        """
        h.set_title(ax, title, titlecolor)
        #  ---------
        result = dict()
        for cat, col in reversed(list(zip(cats, cat_colors))):
            result[cat] = dict()
            vv = variable[factor == cat]
            if len(vv) > 1:
                # # line version
                # result = ax.plot(*h.distribution(vv), color=col, label=cat, linewidth=1)
                # dots version
                result[cat]['scatter'] = ax.scatter(*h.distribution(vv), s=.2, color=col, label=cat)
                # `~matplotlib.collections.PathCollection`
        if legend:
            ax.legend(title=covarname)
        if axes_labels:
            ax.set_xlabel(varname)
        #  ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        #
        return dict(ax=ax, result=result)

    def rocs(ax, variable, factor, cats, cat_colors, cmap, title="distributions by levels", legend=legend):
        """
        """
        h.set_title(ax, title, titlecolor)
        #  ---------
        fpr, tpr, thresh, auroc = rocs0(variable, factor)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.plot([0, 1], [0, 1], color=axescolor, lw=2, linestyle=":")

        if len(cats) > 2:
            for cat, col in reversed(list(zip(cats, cat_colors))):
                lines = ax.plot(fpr[cat], tpr[cat], color=col)
        else:
            lines = ax.plot(fpr, tpr, color=cat_colors[1], label=f"AUC = {round(auroc, 4)}")
            print(f"AUC = {round(auroc, 4)}")

        if legend:
            ax.legend(loc="lower right")
        if axes_labels:
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
        #  ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        #
        result = dict(fpr=fpr, tpr=tpr, thresh=thresh, auroc=auroc, lines=lines)
        return dict(ax=ax, result=result)

    def boxplots(ax, variable, factor, cats, cat_colors, cmap, title="box-plots", horizontal=True, color=color):
        """
        future:
            - violinplots
        """
        h.set_title(ax, title, titlecolor)
        #  ---------

        vvg = variable.groupby(factor)
        data = [vvg.get_group(g) for g in cats if g in vvg.groups.keys()]

        bplot = ax.boxplot(
            data, labels=cats, vert=(not horizontal),
            notch=False,
            #
            patch_artist=True,                              # !!!
            boxprops=dict(color=color, facecolor=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color, marker="|"),
            medianprops=dict(color=color, lw=3),   # (color='white' if color in ['k', 'black'] else 'k', lw=2),
            #
            showmeans=True,
            # meanline=False,
            meanprops=dict(  # color='white' if color in ['k', 'black'] else 'k',
                             marker="d",
                             markeredgecolor=color,
                             markerfacecolor='white' if color in ['k', 'black'] else 'k', markersize=11))

        for patch, color in zip(bplot['boxes'], cat_colors):
            patch.set_facecolor(color)

        if axes_labels:
            ax.set_xlabel(varname)
            ax.set_ylabel(covarname)
        #  ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        #
        result = dict(bplot=bplot)
        return dict(ax=ax, result=result)

    # -----------------------------------------------------
    #  factor ~ factor

    def barchart(ax, title="bar-chart", horizontal=True, align=True, bar_height=bar_height):
        """
        for both factors
        future:
            - legend (bbox_to_anchor, loc, ...)
            - figure size for large amount of levels and long level names
        """
        h.set_title(ax, title, titlecolor)
        #  ---------

        labels = variable.cat.categories.to_list()

        df0 = pd.concat([variable, covariate], axis=1)
        df0g = df0.groupby([variable.name, covariate.name])
        # df0g.agg(len)
        data = df0g.agg(len).unstack(fill_value=0)     # !!!
        data_cum = data.cumsum(axis=1)

        if align:
            data_widths = data.apply(lambda x: x / sum(x), axis=1)
            data_cum_1 = data_cum.apply(lambda x: x / max(x), axis=1)
        else:
            data_widths = data
            data_cum_1 = data_cum

        cats_c = covariate.cat.categories.to_list()
        n_cats_c = len(cats_c)
        # cm = mpl.cm.get_cmap("hsv", n_cats_c)
        colors = plt.colormaps['hsv'](np.linspace(0.1, 0.9, n_cats_c))

        # fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, data_cum_1.max().max())

        for i, (cat, color) in enumerate(zip(cats_c, colors)):
            widths = data_widths.iloc[:, i]
            starts = data_cum_1.iloc[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=bar_height,
                            label=cat, color=color)
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, labels=data.iloc[:, i], label_type='center', color=text_color)

        if legend:
            ax.legend(ncol=len(cats_c), bbox_to_anchor=(0, 1),
                      loc='lower left', fontsize='small')

        #  ---------
        return dict(ax=ax, rects=rects, data=data)

    # -----------------------------------------------------
    #  special

    def blank(ax, variable, factor, cats, cat_colors, cmap, title="", *args, **kwargs):
        """"""
        h.set_title(ax, title, titlecolor)
        #  ---------
        ax.plot()
        ax.axis('off')
        ax.text(
            0.5, 0.5, '',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            color='gray', fontsize=10)
        return dict(ax=ax, result=False)

    def error(ax, title="error"):
        """"""
        h.set_title(ax, title, titlecolor)
        #  ---------
        ax.plot()
        ax.axis('off')
        ax.text(
            0.5, 0.5, 'unavailable',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            color='gray', fontsize=10)
        return dict(ax=ax, result=False)

    # -----------------------------------------------------
    #
    PLOTS = {
        "boxplots": {"plot": boxplots, "name": "box-plots"},
        "cloud": {"plot": cloud, "name": "scatter"},
        "grouped_cloud": {"plot": grouped_cloud, "name": "grouped cloud"},
        "densities": {"plot": densities, "name": "densities"},
        "distr": {"plot": distr, "name": "distributions"},
        "rocs": {"plot": rocs, "name": "rocs"},
        "barchart": {"plot": barchart, "name": "bar chart"},
        "blank": {"plot": blank, "name": ""},
        "error": {"plot": error, "name": "error"},
    }

    # -------------------------------------------------------------------------
    #  plotting procedure

    # -----------------------------------------------------
    #  sizes
    def set_fig(nrows=1, ncols=1):
        nonlocal size
        nonlocal height
        nonlocal width
        nonlocal figsize
        nonlocal figheight
        nonlocal figwidth

        if nrows == 0 or ncols == 0:
            """
            for uneven (custom) figure split into axes
            see  numeric_vs_numeric()  ->  cloud()
            """

            if figsize is None:

                if figheight is None:
                    figheight = size * 2.7 if height is None else height

                if figwidth is None:
                    figwidth = size * 2.7 if width is None else width

                figsize = figwidth, figheight

            fig = plt.figure(figsize=figsize)
            axs = None

        else:

            if figsize is None:

                if figheight is None:
                    height = size if height is None else height
                    figheight = height * nrows + 1     # ? +1 ?

                if figwidth is None:
                    width = size * width_adjust if width is None else width
                    figwidth = width * ncols

                figsize = figwidth, figheight

            fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        # axs = np.reshape(axs, (nrows, ncols))    # unfortunately it's necessary because ...

        return fig, axs

    # -----------------------------------------------------
    #

    def numeric_vs_numeric():
        """"""
        nonlocal result
        nonlocal variable, covariate
        nonlocal df
        nonlocal color
        nonlocal s
        nonlocal alpha
        nonlocal color_data

        fig, _ = set_fig(0)

        # statistics:
        if print_info:  # of course!
            print()
            print(" 2. statistics for processed data")
            print(df_variation)
            print()
            print(df_distribution)

        variable, covariate, color, s, alpha, color_data = \
            sample_and_align(variable, covariate, n_obs=n_obs, shuffle=shuffle, random_state=random_state,
                             color=color, s=s, alpha=alpha, color_data=color_data)

        resc = cloud(fig, title=title)   # fig, axes, result

        h.set_xscale(resc['axes']['ax'], coalesce(xscale, scale))
        h.set_xscale(resc['axes']['ax_histx'], coalesce(xscale, scale))

        h.set_yscale(resc['axes']['ax'], coalesce(yscale, scale))
        h.set_yscale(resc['axes']['ax_histy'], coalesce(yscale, scale))

        result['plot']["cloud"] = {'axes': resc['axes'], 'result': resc['result']}
        fig = resc['fig']

        return fig

    def numeric_vs_factor(num, fac, most_common):
        """"""
        nonlocal result
        nonlocal df0
        nonlocal df
        nonlocal what
        nonlocal cmap
        nonlocal color
        nonlocal s
        nonlocal alpha
        nonlocal color_data

        #  --------------------------------------
        # for potentially processed data    -- but NOT sampled yet (i.e. always all data!)
        df1 = pd.concat([num, fac], axis=1)
        df1agg = df1.groupby([fac.name]).agg([len, np.mean])
        df1agg = df1agg.droplevel(level=0, axis=1).sort_values(by=["len"], ascending=False)

        if data_were_processed:
            df0 = df0.iloc[df.index, :]
            df0agg = df0.groupby([fac.name]).agg(np.mean)
            df0agg.columns = ["mean oryg."]

            df1agg = pd.merge(df1agg, df0agg, left_index=True, right_index=True, how='left')

        # statistics:
        if print_info:  # of course!
            print()
            print(" 2. statistics for processed data")
            print(df_variation)
            print()
            print(df1agg)

        # ---------------------------------------
        #  figure and plots sizes
        what = np.array(what, ndmin=2)
        nrows = what.shape[0]
        ncols = what.shape[1]

        fig, axs = set_fig(nrows, ncols)
        axs = np.reshape(axs, (nrows, ncols))    # unfortunately it's necessary because ...

        cats, cat_colors, cmap = cats_and_colors(fac, most_common, cmap)

        for t in ["boxplots", "blank"]:
            if t in what:
                ax = axs[np.nonzero(what == t)][0]
                try:
                    result['plot'][t] = PLOTS[t]["plot"](ax, num, fac, cats, cat_colors, cmap, PLOTS[t]["name"])
                except Exception as e:
                    print(e)
                    result['plot'][t] = PLOTS["error"]["plot"](ax, PLOTS[t]["name"])

        num, fac, color, s, alpha, color_data = \
            sample_and_align(num, fac, n_obs=n_obs, shuffle=shuffle, random_state=random_state,
                             color=color, s=s, alpha=alpha, color_data=color_data)

        for t in ["grouped_cloud", "densities", "distr", "rocs"]:
            if t in what:
                ax = axs[np.nonzero(what == t)][0]
                try:
                    result['plot'][t] = PLOTS[t]["plot"](ax, num, fac, cats, cat_colors, cmap, PLOTS[t]["name"])
                    # if lines and not isinstance(bins, int):
                    #     for l, c in zip(bins, np.vstack([cmap.colors, cmap.colors[-1]])):
                    #         ax.axvline(l, color=c, alpha=.3)
                except Exception as e:
                    print(e)
                    result['plot'][t] = PLOTS["error"]["plot"](ax, PLOTS[t]["name"])

        result['plot']["agg"] = df1agg

        return fig

    def factor_vs_factor():
        """
        this is very crude version -- TEST IT - FIX IT !!!
        """
        nonlocal result
        nonlocal height
        nonlocal width

        n_levels_covariate = len(covariate.cat.categories)
        longest_variable_level = max(map(lambda s: len(str(s)), variable.cat.categories))
        if height is None:
            height = max(n_levels_variable * bar_height * 1.1, size)
        if width is None:
            width = max(n_levels_covariate * 1. + longest_variable_level * .1, size * width_adjust)

        fig, axs = set_fig(nrows=1, ncols=1)
        ax, rects, data = barchart(axs, title=title)
        # set_axescolor(ax, axescolor)

        result['plot']["barchart"] = ax, rects, data

        # statistics:
        if print_info:  # of course!
            print()
            print(" 2. statistics for processed data")
            print(df_variation)
            print()
            print(data)

        return fig

    # -----------------------------------------------------
    #  core

    if is_factor_y:
        n_levels_variable = len(variable.cat.categories)

        if is_factor_x:

            fig = factor_vs_factor()

        else:

            fig = numeric_vs_factor(covariate, variable, most_common=variable_vc)
    else:

        if is_factor_x:

            fig = numeric_vs_factor(variable, covariate, most_common=covariate_vc)

        else:

            fig = numeric_vs_numeric()

    # -------------------------------------------------------------------------
    #  final

    h.set_figtitle(fig, title, suptitlecolor, suptitlesize)

    fig.tight_layout()
    # plt.show()

    result['plot']["fig"] = fig

    return None if not res else result

# %%
