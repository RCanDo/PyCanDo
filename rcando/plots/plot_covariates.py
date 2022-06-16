# -*- coding: utf-8 -*-
#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

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
      i.e. probaility of each category wrt to value of x.
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

#%%
from .helpers import *

#%% plot_()

def plot_covariates(variable, covariate, data=None,
        varname=None, covarname=None, title=None,
        ignore_index=False,
        factor_threshold=13,
        as_factor_y=None, as_factor_x=None,
        what = ["grouped_cloud", "densities", "boxplots"],
        # Variables modifications (before plotting)
        lower_y=None, upper_y=None, exclude_y=None,
        lower_x=None, upper_x=None, exclude_x=None,
        transform_y=None, transform_x=None,
        lower_t_y=None, upper_t_y=None, exclude_t_y=None,
        lower_t_x=None, upper_t_x=None, exclude_t_x=None,
        #
        bins=21, smooth=.5, # 0 or 1 for OLS (check it!)
        qq_plot=False,            # add qq-plot for  num_vs_num  plot
        sort_levels=True, legend=True, print_levels=True, bar_height=.5,
        n_obs=int(1e4), random_state=None, shuffle=False,        # ???
        # Graphical parameters
        most_common=13,
        figsize=None, figwidth=None, figheight=None, # for the whole figure
        width=None, height=None, size=5, width_adjust=1.2, # for the single plot
        xscale="linear", yscale="linear",
        lines=True,     # not used here yet!
        cmap="Paired",  # for coloring of bars in "hist" and respective points of "agg"
        alpha=None, s=9,   # alpha and size of a data point in a "cloud"
        style=True, color=None, grid=True, titlecolor=None,
        suptitlecolor=None, suptitlesize=1., # multiplier of 15
        #
        tex=False,  # varname & covarname passed in TeX format e.g. "\\hat{Y}" (double`\` needed)
        print_info=True, res=False,
        *args, **kwargs
        ):
    """
    Remarks
    -------
    1. It is assumed that e.g. variable may be passed as pd.Series
    while covariate as a string indicating column of `data`.
    variable and covariate may have different indices while they may be irrelevant
    (to be ignored).
    Thus `ignore_index` is provided which have different meaning from the same parameter of pd.concat():
    here it means that if `ignore_index=True` then we ingore indices of variable and covariate
    and make one index common to both of them;
    It is important for proper aligning of both data series.
    Default value for `ignore_index` is False what means that we pay attention to
    both indices and align two series according to indices values.

    2. Plots are drawn as `variable ~ covariate` i.e.
    `variable` serves as `y` (_dependent_ or _explained_ variable)
    and `covariate` serves as `x` (_independent_ or _explanatory_ variable).
    All parameter names where 'x' or 'y' is used are based on this convention.
    It was preferred to use 'x' or 'y' for its brevity;
    However `variable` and `covariate` are used for the first two parameters
    (instead of 'y' and 'x') to convay their meaning and objective of the whole function:
    explain (via plots) `variable` (`y`) with `covariate` (`x`).

    - `most_common` applied before `sort_levels`
    """
    ##-------------------------------------------------------------------------
    ## loading data

    if isinstance(variable, str):
        variable = data[variable].copy()
    else:
        variable = pd.Series(variable)      # for lists or np.arrays
        if variable.name is None:
            variable.name = "variable"

    if varname is None:
        varname = coalesce(variable.name, "Y")


    if isinstance(covariate, str):
        covariate = data[covariate].copy()
    else:
        covariate = pd.Series(covariate)      # for lists or np.arrays
        if covariate.name is None:
            covariate.name = "covariate"

    if covarname is None:
        covarname = coalesce(covariate.name, "X")

    if varname == covarname:
        covarname += "_0"

    #!!! index is ABSOLUTELY CRITICAL here !!!
    if ignore_index:
        covariate.index = variable.index

    ##-----------------------------------------------------
    ## info on raw data
    df0 = pd.concat([variable, covariate], axis=1)
    df0.columns = [varname, covarname]

    df0_info = info(df0, what = ["dtype", "oks", "oks_ratio", "nans_ratio", "nans"])
    if print_info:
        print(" 1. info on raw data")
        print(df0_info)

    variable = df0[varname]
    covariate = df0[covarname]

    del df0

    ##-------------------------------------------------------------------------
    ## preparing data

    if not lower_y is None:
        variable = variable[variable >= lower_y]
        #lower_y_str = f"{lower_y} <= "
    if not upper_y is None:
        variable = variable[variable <= upper_y]
        #upper_y_str = f" <= {upper_y}"
    if not exclude_y is None:
        variable = variable[~ variable.isin(flatten([exclude_y]))]

    if not lower_x is None:
        covariate = covariate[covariate >= lower_x]
    if not upper_x is None:
        covariate = covariate[covariate <= upper_x]
    if not exclude_x is None:
        covariate = covariate[~ covariate.isin(flatten([exclude_x]))]


    is_factor_y, variable, variable_vc  = to_factor(variable, as_factor_y, most_common=most_common, factor_threshold=factor_threshold)
    is_factor_x, covariate, covariate_vc = to_factor(covariate, as_factor_x, most_common=most_common, factor_threshold=factor_threshold)

    df0 = pd.concat([variable, covariate], axis=1)
    df0.columns = [varname, covarname]
    df0.dropna(inplace=True)
    df0.index = range(df0.shape[0])
    # or df = df.reset_index() # but it creates new column with old index -- mess

    variable = df0[varname]
    covariate = df0[covarname]

    df = df0
    # df0 -- unprocessed data, i.e. only to_factor(...) and .dropna()
    # df -- potentially processed data

    ##-----------------------------------------------------
    ## transforms

    transform_y = coalesce(transform_y, False)      # function always have name so it's spurious
    if transform_y and not is_factor_y:
        if isinstance(transform_y, bool):
            variable, transform_y = power_transformer(variable)
        else:
            transform_y.__name__ = coalesce(transform_y.__name__, "T_y")
            variable = pd.Series(transform_y(variable))

        if not lower_t_y is None:
            variable = variable[variable >= lower_t_y]
        if not upper_t_y is None:
            variable = variable[variable <= upper_t_y]
        if not exclude_t_y is None:
            variable = variable[~ variable.isin(flatten([exclude_t_y]))]

    transform_x = coalesce(transform_x, False)      # function always have name so it's spurious
    if transform_x and not is_factor_x:
        if isinstance(transform_x, bool):
            covariate, transform_x = power_transformer(covariate)
        else:
            transform_x.__name__ = coalesce(transform_x.__name__, "T_x")
            covariate = pd.Series(transform_x(covariate))

        if not lower_t_x is None:
            covariate = covariate[covariate >= lower_t_x]
        if not upper_t_x is None:
            covariate = covariate[covariate <= upper_t_x]
        if not exclude_t_x is None:
            covariate = covariate[~ covariate.isin(flatten([exclude_t_x]))]

    ## !!! TODO !!!  make it robust on (*): when False passed data_were_processed=True -- bad!
    transforms = [transform_y, lower_t_y, upper_t_y, exclude_t_y, transform_x, lower_t_x, upper_t_x, exclude_t_x]
    if any(transforms):
        df = pd.concat([variable, covariate], axis=1)
        df.columns = [varname, covarname]
        df.dropna(inplace=True)
        df.index = range(df.shape[0])
        # or df = df.reset_index() # but it creates new column with old index

        variable = df[varname]
        covariate = df[covarname]

        data_were_processed = True     # (*)
    else:
        data_were_processed = False    # (*)

    print(data_were_processed)

    ##-----------------------------------------------------
    ## statistics for processed data
    df_variation = summary(df,
        what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    df_distribution = summary(df,
        what=["range", "iqr", "mean", "median", "min", "max", "negatives", "zeros", "positives"])

    ## ...

    ##-----------------------------------------------------
    ## title

    if title is None:

        title = make_title(varname, lower_y, upper_y, transform_y, lower_t_y, upper_t_y, tex) + \
                " ~ " + \
                make_title(covarname, lower_x, upper_x, transform_x, lower_t_x, upper_t_x, tex)

    ## ----------------------------------------------------
    ## !!! result !!!

    result = { "title" : title,
        "df0" : df0,  # unprocessed
        "df" : df,    # processed
        "info" : df0_info,
        "variation": df_variation,
        "distribution": df_distribution,
        }  # variable after all prunings and transformations

    ##---------------------------------------------------------------------------------------------
    ## plotting

    ##-------------------------------------------------------------------------
    ## style affairs

    N = len(variable) if not n_obs else min(len(variable), int(n_obs))
    style, color, grid, suptitlecolor, titlecolor, alpha = \
        style_affairs(style, color, grid, suptitlecolor, titlecolor, alpha, N)

    ##-------------------------------------------------------------------------
    ## helpers
    ## TODO: implement them as decorators !

    ##-------------------------------------------------------------------------
    ## plot types

    def qq(covariate, variable):
        qqx = [covariate.quantile(q/10) for q in range(11)]
        qqy = [variable.quantile(q/10) for q in range(11)]
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
        scatter = ax.scatter(covariate, variable, s=s, color=color, alpha=alpha)
        if qq_plot:
            qqx, qqy = qq(covariate, variable)
            ax.plot(qqx, qqy, color=mpl.colors.to_rgba(color, .5), marker="*")
        else:
            qqx, qqy = None, None

        # now determine nice limits by hand:
        #binwidth = 0.25
        #xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        #lim = (int(xymax/binwidth) + 1) * binwidth

        #bins = np.arange(-lim, lim + binwidth, binwidth)
        histx = ax_histx.hist(covariate, bins=bins, color=mpl.colors.to_rgba(color, .6))
        histy = ax_histy.hist(variable,  bins=bins, color=mpl.colors.to_rgba(color, .6), orientation='horizontal')

        return ax, scatter, histx, histy, qqx, qqy

    def smoother(ax):
        """helper for cloud()
        lowess trend of variable vs covariate
        """
        xx = np.linspace(min(covariate), max(covariate), 100)
        #print(xx)
        smoothed = sm.nonparametric.lowess(exog=covariate, endog=variable,
            xvals=xx,
            frac=smooth)
        #print(smoothed)

        ax.plot(xx, smoothed, c="r" if color != 'r' else 'k')

        return ax, xx, smoothed

    def cloud(fig, title="scatter"):
        """
        On how to get side histograms
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
        """
        # set_title(ax, title, titlecolor)
        # ## ---------

        # definitions for the axes
        left, width = 0.1, 0.7
        bottom, height = 0.1, 0.7
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.145]
        rect_histy = [left + width + spacing, bottom, 0.149, height]

        # start with a square Figure
        #fig = plt.figure(figsize=(8, 8))

        ax = fig.add_axes(rect_scatter)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # use the previously defined function
        ax, scatter, histx, histy, qqx, qqy = scatter_hist(ax, ax_histx, ax_histy)
        ax, xx, smoothed = smoother(ax)

        #plt.show()

        ## ---------
        if grid:
            if not isinstance(grid, bool):
                ax.grid(**grid)
                ax_histx.grid(**grid)
                ax_histy.grid(**grid)
        else:
            ax.grid(visible=False, axis="both")
            ax_histx.grid(visible=False, axis="both")
            ax_histy.grid(visible=False, axis="both")

        return fig, ax, scatter, ax_histx, histx, ax_histy, histy, qqx, qqy

    #%%
    def cats_and_colors(factor, most_common):
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
        if not most_common is None:
            cats = [cat for cat in cats if cat in most_common]
        #
        cat_colors = plt.colormaps['hsv'](np.linspace(0.1, 0.9, len(cats)))
        cmap = ListedColormap(cat_colors)
        return cats, cat_colors, cmap

    def grouped_cloud(ax, variable, factor, cats, cat_colors, cmap, title="grouped cloud", sort=sort_levels):
        """
        """
        set_title(ax, title, titlecolor)
        ## ---------

        dff = pd.concat([variable, factor, factor.cat.codes], axis=1)
        dff.columns = [variable.name, factor.name, "codes"]
        if sort:
            dff.sort_values(by="codes", ignore_index=True, inplace=True)

        dff = dff[dff[factor.name].isin(cats)]     # sorting is retained !

        scatter = ax.scatter(dff[variable.name], dff.index, c=dff['codes'], cmap=cmap, s=s, alpha=alpha)

        ## ---------
        set_grid(ax, off="both", grid=grid)

        return ax, scatter, cat_colors, dff

    def densities(ax, variable, factor, cats, cat_colors, cmap, title="densities by levels", legend=legend):
        """
        """
        set_title(ax, title, titlecolor)
        ## ---------

        for cat, col in reversed(list(zip(cats, cat_colors))):
            vv = variable[factor == cat]
            if len(vv) > 1:
                try:
                    density = gaussian_kde(vv.astype(float))
                except Exception:
                    density = gaussian_kde(vv)
                xx = np.linspace(min(vv), max(vv), 200)
                ax.plot(xx, density(xx), color=col, label=cat)
        if legend:
            ax.legend()

        ## ---------
        set_grid(ax, off="both", grid=grid)

        return ax, xx, density

    def boxplots(ax, variable, factor, cats, cat_colors, cmap, title="box-plots", horizontal=True, color=color):
        """
        TODO:
            - violinplots
       """
        set_title(ax, title, titlecolor)
        ## ---------

        vvg = variable.groupby(factor)
        data = [vvg.get_group(g) for g in cats if g in vvg.groups.keys()]

        bplot = ax.boxplot(data, labels=cats, vert=(not horizontal),
            notch=False,
            #
            patch_artist = True,                              #!!!
            boxprops = dict(color=color, facecolor=color),
            whiskerprops = dict(color=color),
            capprops = dict(color=color),
            flierprops = dict(color=color, markeredgecolor=color, marker="|"),
            medianprops = dict(color=color, lw=3),   #(color='white' if color in ['k', 'black'] else 'k', lw=2),
            #
            showmeans = True,
            #meanline = False,
            meanprops = dict(#color='white' if color in ['k', 'black'] else 'k',
                             marker="d",
                             markeredgecolor=color,
                             markerfacecolor='white' if color in ['k', 'black'] else 'k', markersize=11)
            )

        for patch, color in zip(bplot['boxes'], cat_colors):
            patch.set_facecolor(color)

        ## ---------
        set_grid(ax, off="both", grid=grid)

        return ax, bplot


    def barchart(ax, title="bar-chart", horizontal=True, align=True, bar_height=bar_height):
        """
        for both factors
        TODO:
            - legend (bbox_to_anchor, loc, ...)
            - figure size for large amount of levels and long level names
        """
        set_title(ax, title, titlecolor)
        ## ---------

        labels = variable.cat.categories.to_list()

        df0 = pd.concat([variable, covariate], axis=1)
        df0g = df0.groupby([variable.name, covariate.name])
        #df0g.agg(len)
        data = df0g.agg(len).unstack(fill_value=0)     #!!!
        data_cum = data.cumsum(axis=1)

        if align:
            data_widths = data.apply(lambda x: x/sum(x), axis=1)
            data_cum_1 = data_cum.apply(lambda x: x/max(x), axis=1)
        else:
            data_widths = data
            data_cum_1 = data_cum


        cats_c = covariate.cat.categories.to_list()
        n_cats_c = len(cats_c)
        #cm = mpl.cm.get_cmap("hsv", n_cats_c)
        colors = plt.colormaps['hsv'](np.linspace(0.1, 0.9, n_cats_c))

        #fig, ax = plt.subplots(figsize=(9.2, 5))
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
            ax.bar_label(rects, labels=data.iloc[:, i] , label_type='center', color=text_color)

        if legend:
            ax.legend(ncol=len(cats_c), bbox_to_anchor=(0, 1),
                      loc='lower left', fontsize='small')

        return ax, rects, data

    def error(ax, title="error"):
        """"""
        set_title(ax, title, titlecolor)
        ## ---------
        ax.plot()
        ax.axis('off')
        ax.text(0.5, 0.5, 'unavailable',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            color='gray', fontsize=10)
        return ax, False

    def blank(ax, title="", *args, **kwargs):
        """"""
        set_title(ax, title, titlecolor)
        ## ---------
        ax.plot()
        ax.axis('off')
        ax.text(0.5, 0.5, 'unavailable',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            color='gray', fontsize=10)
        return ax, False


    ##-----------------------------------------------------
    ##
    PLOTS = {
        "boxplots":  {"plot": boxplots, "name": "box-plots"},
        "cloud":     {"plot": cloud, "name": "scatter"},
        "grouped_cloud": {"plot": grouped_cloud, "name": "grouped cloud"},
        "densities": {"plot": densities, "name": "densities"},
        "barchart":  {"plot": barchart, "name": "bar chart"},
        "error":     {"plot": error, "name": "error"},
        "blank":     {"plot": blank, "name": ""},
        }


    ##-------------------------------------------------------------------------
    ## plotting procedure

    ##-----------------------------------------------------
    ## sizes
    def set_fig(nrows=1, ncols=1):
        nonlocal size
        nonlocal height
        nonlocal width
        nonlocal figsize
        nonlocal figheight
        nonlocal figwidth

        if nrows==0 or ncols==0:
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
                    figheight = height * nrows + 1     #? +1 ?

                if figwidth is None:
                    width = size * width_adjust if width is None else width
                    figwidth = width * ncols

                figsize = figwidth, figheight

            fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        #axs = np.reshape(axs, (nrows, ncols))    # unfortunately it's necessary because ...

        return fig, axs

    ##-----------------------------------------------------
    ##
    def sample(xx, yy):
        """  """
        df = pd.concat([xx, yy], axis=1, ignore_index=False)
        if n_obs and n_obs < len(df):
            df = df.sample(int(n_obs), ignore_index=False, random_state=random_state)
        if shuffle:
            df = df.sample(frac=1, ignore_index=True, random_state=random_state)
        xx = df[xx.name]
        yy = df[yy.name]
        return xx, yy


    def numeric_vs_factor(num, fac, most_common):
        """"""
        nonlocal result
        nonlocal df0
        nonlocal df
        nonlocal what

        ## --------------------------------------
        # for potentially processed data    -- but NOT sampled yet (i.e. always all data!)
        df1 = pd.concat([num, fac], axis=1)
        df1agg = df1.groupby([fac.name]).agg([len, np.mean])
        df1agg = df1agg.droplevel(level=0, axis=1).sort_values(by=["len"], ascending=False)

        if data_were_processed:
            df0 = df0.iloc[df.index, :]
            df0agg = df0.groupby([fac.name]).agg(np.mean)
            df0agg.columns = ["mean oryg."]

            df1agg = pd.merge(df1agg, df0agg, left_index=True, right_index=True, how='left')

        ##---------------------------------------
        ## figure and plots sizes
        what = np.array(what, ndmin=2)
        nrows = what.shape[0]
        ncols = what.shape[1]


        fig, axs = set_fig(nrows, ncols)
        axs = np.reshape(axs, (nrows, ncols))    # unfortunately it's necessary because ...

        cats, cat_colors, cmap = cats_and_colors(fac, most_common)

        for t in ["boxplots", "blank"]:
            if t in what:
                ax = axs[np.nonzero(what==t)][0]
                try:
                    result[t] = PLOTS[t]["plot"](ax, num, fac, cats, cat_colors, cmap, PLOTS[t]["name"])
                    set_xscale(ax, xscale)
                except Exception as e:
                    print(e)
                    result[t] = PLOTS["error"]["plot"](ax, PLOTS[t]["name"])

                    #!
                    #result["boxplots"] = boxplots(axs[2], num, fac, cats, cat_colors, cmap)

        num, fac = sample(num, fac)

        for t in ["grouped_cloud", "densities"]:
            if t in what:
                ax = axs[np.nonzero(what==t)][0]
                try:
                    result[t] = PLOTS[t]["plot"](ax, num, fac, cats, cat_colors, cmap, PLOTS[t]["name"])
                    # if lines and not isinstance(bins, int):
                    #     for l, c in zip(bins, np.vstack([cmap.colors, cmap.colors[-1]])):
                    #         ax.axvline(l, color=c, alpha=.3)
                    set_xscale(ax, xscale)
                except Exception as e:
                    print(e)
                    result[t] = PLOTS["error"]["plot"](ax, PLOTS[t]["name"])

                    #!
                    #result["grouped_cloud"] = grouped_cloud(axs[0], num, fac, cats, cat_colors, cmap)
                    #result["densities"] = densities(axs[1], num, fac, cats, cat_colors, cmap)

        # statistics:
        if print_info:  # of course!
            print()
            print(" 2. statistics for processed data")
            print(df_variation)
            print(df1agg)

        result["agg"] = df1agg

        return fig


    def factor_vs_factor():
        """
        TODO: this is very crude version -- TEST IT - FIX IT !!!
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
        result["barchart"] = ax, rects, data

        # statistics:
        if print_info:  # of course!
            print()
            print(" 2. statistics for processed data")
            print(df_variation)
            print()
            print(data)

        return fig


    def numeric_vs_numeric():
        """"""
        nonlocal result
        nonlocal variable, covariate

        fig, _ = set_fig(0)

        # statistics:
        if print_info:  # of course!
            print()
            print(" 2. statistics for processed data")
            print(df_variation)
            print()
            print(df_distribution)

        variable, covariate = sample(variable, covariate)

        fig, ax, scatter, ax_histx, histx, ax_histy, histy, qqx, qqy = cloud(fig, title=title)

        set_xscale(ax, xscale)
        set_xscale(ax_histx, xscale)

        set_yscale(ax, yscale)
        set_yscale(ax_histy, yscale)

        result["cloud"] = ax, scatter, ax_histx, histx, ax_histy, histy, qqx, qqy

        return fig

    ##-----------------------------------------------------
    ## core
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


    ##-------------------------------------------------------------------------
    ## final

    if suptitlecolor:
        fig.suptitle(title, fontweight='normal', color=suptitlecolor, fontsize=15 * suptitlesize)
    else:
        fig.suptitle(title, fontweight='normal', fontsize=15 * suptitlesize)
    fig.tight_layout()
    plt.show()

    result["fig"] = fig

    return None if not res else result

#%%
