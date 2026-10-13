#! python3
# -*- coding: utf-8 -*-
"""
---
title: Variable (y) vs covariate (x)
version: 1.0
type: module
keywords: [plot, covariates, ]
description: |
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
import utils.plots.helpers as h

from typing import Union, Sequence, List, Tuple, Literal, Callable, Optional

Numeric = Sequence[float]
Factor = Sequence[str]
Variable = Union[Numeric, Factor]
Transformation = Callable[[Numeric], Numeric]
Aggregate = Callable[[Numeric], float]
PlotType = Literal["cloud", "grouped_cloud", "densities", "boxplots", "distr", "barchart", "blank"]
Scale = Literal["linear", "log"]


# %% plot_()
def plot_covariates(
        variable: str | Variable,
        covariate: str | Variable,
        data: Optional[pd.DataFrame] = None,
        what: List[PlotType] | List[List[PlotType]] = ["grouped_cloud", "densities", "boxplots"],
        varname: Optional[str] = None, covarname: Optional[str] = None,
        title: Optional[str] = None, title_suffix: Optional[str] = None,
        ignore_index: bool = False,
        # Variables modifications (before plotting)
        n_obs: Optional[int] = int(1e4),
        shuffle: bool = False, random_state: Optional[int] = None,         # ???
        most_common: int = 13, print_levels: bool = True,
        factor_threshold: int = 13,
        as_factor_y: Optional[bool] = None, as_factor_x: Optional[bool] = None,
        #
        lower_y: Optional[float] = None, upper_y: Optional[float] = None,
        exclude_y: Optional[Sequence[float] | float] = None,
        lower_x: Optional[float] = None, upper_x: Optional[float] = None,
        exclude_x: Optional[Sequence[float] | float] = None,
        transform_y: Optional[bool | Transformation] = None,
        transform_x: Optional[bool | Transformation] = None,
        lower_t_y: Optional[float] = None, upper_t_y: Optional[float] = None,
        exclude_t_y: Optional[Sequence[float] | float] = None,
        lower_t_x: Optional[float] = None, upper_t_x: Optional[float] = None,
        exclude_t_x: Optional[Sequence[float] | float] = None,
        # Graphical parameters
        # # data point
        color: Optional[str] = None, s: int | str = 9, alpha: Optional[float | str] = None,
        brightness: Optional[float] = None,  # not used yet
        # # Parameters for various types of plots
        bins: int | List[float] = 21, smooth: float = .5,
        qq_plot: bool = False,       # add qq-plot for  num ~ num  plot
        sort_levels: bool = True, legend: bool = True, axes_labels: bool = True,
        bar_height: float = .5,
        # # Figure and axes parameters
        # ## for the whole figure
        figsize: Optional[Tuple[float, float]] = None,
        figwidth: Optional[float] = None, figheight: Optional[float] = None,
        # ## for the single axis (subplot)
        width: Optional[float] = None, height: Optional[float] = None, size: float = 5, width_adjust: float = 1.2,
        scale: Scale = "linear", xscale: Optional[Scale] = None, yscale: Optional[Scale] = None,
        lines: bool = True,  # not used here yet!
        cmap: str = "ak01",  # for coloring of bars in "hist"
        #
        style: Optional[str | bool] = True, grid: Optional[bool | dict] = True,
        axescolor: Optional[str] = None, titlecolor: Optional[str] = None,
        suptitlecolor: Optional[str] = None, suptitlesize: float = 1.,  # multiplier of 15
        #
        tex: bool = False,  # varname & covarname passed in TeX format e.g. "\\hat{Y}" (double`\` needed)
        print_info: bool = True, res: bool = False,
        *args, **kwargs
) -> Optional[dict]:
    """
    What is the dependence of a variable (y) on the covariate (x)?
    Symbolically `y ~ x` or `y vs x`.
    There are 4 cases:
        (y is numeric | categorical) ~ (x is numeric | categorial)
        (y is num | cat) ~ (x is num | cat)
    but cat ~ num is always turned to num ~ cat (i.e. variable and covariate are swapped in such case)
    as it's more straitforward to show dependency this way (however, see todo).
    In this case (num ~ cat) there are 3 main types of plots:
    - grouped cloud,
    - densities (separate density of y for each level of x),
    - boxplots (or each level of x);
    num ~ num is just an ordinary scatter-plot,
    while cat ~ cat is a bar-chart.
    As for the plot_variable() the idea was to make it fully automated but also very flexible:
    lots of intelligent preprocessing, fully parameterised but with sensible defaults.

    Basic params
    ------------
    variable: str | Variable (any Sequence of data);
        if str then it indicates column of `data`;
        else any Sequence of data (like pd.Series or just list) to be plotted;
        referenced for brevity as `y`;
        we plot `y ~ x` i.e. variable on y axis as dependent on covariate – x axis;
    covariate: str | Variable (any Sequence of data);
        as for `variable`
        referenced for brevity as `x`.
    data: pd.DataFrame = None
        if None then `variable` must be Sequence of data (e.g. pd.Series) with data of interest;
    what: List[PlotType] | List[List[PlotType]] = [["grouped_cloud", "densities", "boxplots"]];
        This parameter is only considered for num ~ cat case (and cat ~ num which is turned to num ~ cat).
        list of lists (or just list which is internally changed to `list(what)`).
        It reflects the design of the whole final figure where
        each sublist represents one row of plots (axes) within a figure
        and each element of a sublist is the name of the respective plot type
        which will be rendered in a respective subplot (axis) of the figure.
        Thus each sublist should be of the same length (if not there will be blank subplots);
        possible values of the elements of the sublist (types of the plots) are:
            "cloud", "grouped_cloud", "densities", "boxplots", "distr", "barchart", "blank" (for empty subplot);
    varname: Optional[str] = None;
        variable name to be used in title, etc.;
        if None then taken from .name attr of `variable` if exists, and if not then set to 'Y';
    covarname: Optional[str] = None;
        variable name to be used in title, etc.;
        if None then taken from .name attr of `variable` if exists, and if not then set to 'Y';
    title: Optional[str] = None,
        title of a plot; if None then generated automaticaly;
    title_suffix: Optional[str] = None,
        makes it possible to distinguish plots for which automatic title would be the same;
    ignore_index: bool = False,
        if True then `y` and `x` will not be aligned wrt their index (if exists) but only positionally,
        i.e. as they are passed;
        otherwise (default) index is taken into account (if exists);

    Variable modifications (before plotting)
    ----------------------------------------
    n_obs: Optional[int] = int(1e4),
        if not None then maximum nr of observations to be sampled from variable before plotting
        "grouped_cloud", "densities", "distr";
        if None whole data will be plotted (what is usually not sensible for very large data).
    shuffle: bool = False;
        shuffle data before plotting -- useful only for "grouped_cloud" plot in case
        when data are provided in clumps with different statistical properties;
        shuffling helps to spot distribution features of the whole data.
    random_state: Optional[int] = None,
        passed to numpy random generator for reproducibility in case of
        `n_obs` is not None or shuffle is True;
    most_common: int = 13,
        if None all bars for all factor levels will be plotted;
        hence using None is dangerous if not sure how many levels there are;
        it's better to set big integer but no bigger then 100;
        otherwise plot may not be rendered at all if there are thousands of levels;
    print_levels: bool = True,
        do print (in terminal or console) levels of a factor with their counts?

    factor_threshold: int = 13,
        if mumeric variable (or covariate) has less distinct values than `factor_threshold`
        it will be treaded as categorical variable (factor);
        this will happen only if `as_factor_y` or `as_factor_x` are left None;
    as_factor_y: Optional[bool] = None,
        if True variable will be treated as categorical (factor);
        if False then numeric variable is never turned to factor
        (even if it has less distinct values than `factor_threshold`);
        if left None then it is turned into factor iff it has less unique values than `factor_threshold`;
    as_factor_x: Optional[bool] = None,
        the same as `as_factor_y` but for `x` (covariate);

    lower_y: Optional[float] = None;
        lower limit of `variable` (y) to be plotted; inclusive !
        if None then `lower_y == min(variable)`
    upper_y: Optional[float] = None;
        upper limit of `variable` (y) to be plotted; inclusive !
        if None then `upper_y == max(variable)`
    exclude_y: Optional[Sequence[float] | float] = None;
        values to be excluded from `variable` (y) before plotting;
    transform_y: Optional[bool | Transformation] = None;
        what transformation of `variable` (y) to use (if any) before plotting;
        if None or False no transformation is used;
        if True then Yeo-Johnson transformation is used with automatic parameter;
        if function (Callable) is passed then this function is used;

    lower_x: Optional[float] = None;
        as `lower_y` but for `covariate` (x);
    upper_x: Optional[float] = None;
        as `upper_y` but for `covariate` (x);
    exclude_x: Optional[Sequence[float] | float] = None;
        as `exclude_y` but for `covariate` (x);
    transform_x: Optional[bool | Transformation] = None;
        as `transform_y` but for `covariate` (x);

    upper_t_y: Optional[float] = None;
        upper limit of transformed `variable` (y) to be plotted; inclusive !
        if None then `upper_t_y == max(transform_y(variable))`;
    lower_t_y: Optional[float] = None;
        lower limit of transformed `variable` (y) to be plotted; inclusive !
        if None then `lower == min(transformed_y(variable))`;
    exclude_t_y: Optional[Sequence[float] | float] = None;
        values to be excluded from transformed `variable` (y) before plotting;

    lower_t_x: Optional[float] = None;
        as `lower_t_y` but for `covariate` (x);
    upper_t_x: Optional[float] = None;
        as `upper_t_y` but for `covariate` (x);
    exclude_t_x: Optional[Sequence[float] | float] = None;
        as `exclude_t_y` but for `covariate` (x);

    Graphical parameters
    --------------------
    # Data point
    color: Optional[str] = None,
        color of lines and points for 'cloud', 'boxplot', 'density' and 'distr';
        if None then set automatically according to `style`: "yellow" for style "black_background", else to "black";
        if it is str but not a valid color name and `data` is not None
        then it is interpreted as a name of a column in data
        where color values are stored (may be different for each data point);
    s: int | str = 9,
        size of a data point in a "cloud"
        if str and `data` is not None then it is interpreted as a name of a column in data
        where sizes of data point are stored;
    alpha: Optional[float | str] = None,
        float between 0 and 1; for points of "cloud" only;
        if None then set automatically according to `n_obs`;
        if str and `data` is not None then it is interpreted as a name of a column in data
        where alphas for data points are stored;
    brightness: Optional[float] = None,
        not used yet

    # Parameters for various types of plots

    bins: int | List[float] = 21,
        int or list of boarders of bins (passed to ax.hist(...))
        how many or what bins (groups) for "hist";
    smooth: float = .5,
        For num ~ num plot (cloud of points) there is always plotted a lowess trend (of variable vs covariate);
        this param tells how smooth it should be: the higher the value the smoother is the trend estimation.
        Between 0 and 1: the fraction of the data used when estimating each y-value.
        0 means 'turn it off' – no lowess line.
    qq_plot: bool = False,
        do add qq-plot for  num_vs_num  plot?
    sort_levels: bool = True,
        do sort levels of a factor (alphabetically) before plotting?
    legend: bool = True,
        do draw legend on a plot?
    axes_labels: bool = True,
        do give labels to axes?
    bar_height: float = .5,
        only for barchart (factor ~ factor);

    # Figure and axes parameters

    # ## for the whole figure
    If `figsize`, `figwidth`, `figheight` are None then they are calculated based on sizes of a single subplot (axis)
    which are set directly via `width` and `height` or indirectly from `size` and `width_adjust` params.
    Otherwise `figsize` overrides `figwidth`, `figheight` and they in turn override all the other size params.
    Notice that "the other" params, especially `size` and `width_adjust` do not work perfectly
    when translating them into figure size so some experimantation may be necessary to achive desired look of a plot.

    figsize: Optional[Tuple[float, float]] = None,
        (width, height) passed to `plt.figure(figsize=.)`;
    figwidth: Optional[float] = None,
        if `figsize` is None then it is created from `figwidth` and `figheight`;
    figheight: Optional[float] = None,
        if `figsize` is None then it is created from `figwidth` and `figheight`;

    # ## for the single axis (subplot)

    width: Optional[float] = None,
        width of a single subplot (axis);
    height: Optional[float] = None,
        height of a single subplot (axis);
    size: float = 5,
        if `width` or `height` are None then they are calculated from `size` and `width_adjust`:
            height = size
            width = size * width_adjust
    width_adjust: float = 1.2,
        see help on `size`;

    scale: Scale = "linear",
        scale of axes of the subplots of type "cloud", "grouped_cloud", "densities", "distributions", "box_plots";
        possible values are "linear" and "log";
    xscale: Optional[Scale] = None,
        scale type of x-axis for "cloud" plot (num ~ num); overwrites `scale`;
    yscale: Optional[Scale] = None,
        scale type of y-axis for "cloud" plot (num ~ num); overwrites `scale`;

    lines: bool = True;
        not used yet; ...
    cmap: str = "ak01",
        color map name;
        for coloring of bars in "hist";
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py
        or dir(matplotlib.pyplot.cm) for list of all available color maps;
        see https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#creating-colormaps-in-matplotlib
        on how to create and register ListedColormaps.

    #
    style: Optional[str | bool] = True,
        if True takes all the graphic parameters set externally (uses style from environment);
        if False then is set to "dark_background";
        str must be a name of one of available styles: see `plt.style.available`.
    grid: Optional[bool | dict] = True,
        if False then no grid is plotted (regrdless of style);
        if True then grid is plotted as ascribed to given style;
        in case of "black_background" it is dict(color='gray', alpha=.3)
        if dict then values from this dict will be applied as described in
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.grid.html
        Most general form (example values):
        { 'alpha': 1.0,  'color': '#b0b0b0',  'linestyle': '-',  'linewidth': 0.8  }
    axescolor: Optional[str] = None,
        color of the main axes (or rather the frame) of the subplot;
        if None then set automatically according to style;
    titlecolor: Optional[str] = None,
        color of axis titles;
        if None then set automatically according to style;
    suptitlecolor: Optional[str] = None,
        color of the whole title plot (fig.suptitle)
        if None then set automatically according to style;
    suptitlesize: float = 1.,
        multiplier of 15 for the whole title plot (fig.suptitle)
    #
    tex: bool = False,
        informs if the varname & covarname were passed in TeX format e.g. "\\hat{Y}" (double`\` needed)
        or how '_' should be interpreted (as part of name or TeX subscript);
    print_info: bool = True,
        print df.info(variable) (after all transformations)
    res: bool = False,
        do return result of all the calculations?
        default is False and then None is returned;
        otherwise (if True) dictionary is returned with the following structure:

    Returns
    -------
    If required (`ret=True`) it is dictionary "of everything":

        result = {
            "title": title,
            "df0": df0,         # unprocessed pd.DataFrame([variable, covariate])
            "df": df,           # processed   "  (after all clippings and transformations)
            "info": df0_info,                   # info on raw data
            "variation": df_variation,          # statistics for processed data
            "distribution": df_distribution,    # more statistics for processed data
            "plot": {
                "cloud": {
                    "axes": <Axes objects>,
                    "result": <data plotted>
                    }
                <
                  other subplot objects under respective keys as called for by `what`
                  with keys like "axes", "result", "agg" ... (depending on the plot type);
                >
                ...
                "fig": <Figure of a plot>
                }
        }

    Remarks
    -------
    1. Plots are drawn as `variable ~ covariate` i.e.
    `variable` serves as `y` (_dependent_ or _explained_ variable)
    and `covariate` serves as `x` (_independent_ or _explanatory_ variable).
    All parameter names where 'x' or 'y' is used are based on this convention.
    It was preferred to use 'x' or 'y' for its brevity;
    However, `variable` and `covariate` are used for the first two parameters
    (instead of 'y' and 'x') to convey their meaning and objective of the whole function:
    explain (via plots) `variable` (`y`) with `covariate` (`x`).

    2.
    - `style=True` by default what means using style set up externally
      and it is assumed to be set to `plt.style.use('dark_background')`;
    - All default graphic parameters are set for best fit with 'dark_background' style.
    - Unfortunately changing styles is not fully reversible, i.e.
      some parameters (`plt.rcParams`) stays changed after reverting style;
      (eg. grid stays white after 'ggplot',
      the same for title colors after 'black_background', etc);
      Just because of this there are some parameters like `color`, `grid`, `titlecolor`
      to set up their values by hand in case of unwanted "defaults".

    3. It is assumed that `variable` may be passed as `pd.Series`
    while `covariate` as a string indicating column of `data` (and vice versa).
    `variable` and `covariate` may have different indices while they may be irrelevant
    (to be ignored).
    Thus `ignore_index` is provided but it has different meaning from the same parameter of `pd.concat()`:
    here it means that if `ignore_index=True` then we ignore indices of `variable` and `covariate`
    and make one index common to both of them based solely on the elements order
    (thus number of elements must be the same in both series).
    It is critical for proper aligning of both data series.
    Default value for `ignore_index` is False what means that we pay attention to
    both indices and align two series according to indices values (like in pd.concat()).

    4. `most_common` applied before `sort_levels`.
    """
    # ---------------------------------------------------------------------------------------------
    #  preparations

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
    transforms = [transform_y, lower_t_y, upper_t_y, exclude_t_y, transform_x, lower_t_x, upper_t_x, exclude_t_x]
    if any(transforms):
        df = pd.concat([variable, covariate], axis=1)
        df.columns = [varname, covarname]
        df.dropna(inplace=True)

        variable = df[varname]
        covariate = df[covarname]

        data_were_processed = True
    else:
        data_were_processed = False

    # -----------------------------------------------------
    #  statistics for processed data

    df_variation = udf.summary(
        df, what=["oks", "uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"])

    df_distribution = udf.summary(
        df, what=["range", "iqr", "mean", "median", "min", "max", "negatives", "zeros", "positives"])

    # -------------------------------------------------------------------------
    #  title

    if title is None:

        title = h.make_title(varname, lower_y, upper_y, transname_y, lower_t_y, upper_t_y, tex) + \
            " ~ " + \
            h.make_title(covarname, lower_x, upper_x, transname_x, lower_t_x, upper_t_x, tex)
        # the same for  numeric ~ factor  and  factor ~ numeric

    if title_suffix:
        title = title + title_suffix

    # -------------------------------------------------------------------------
    #  result

    result = {
        "title": title,
        "df0": df0,  # unprocessed
        "df": df,    # processed
        "info": df0_info,
        "variation": df_variation,
        "distribution": df_distribution,  # data after all prunings and transformations
        "plot": dict()
    }

    # ---------------------------------------------------------------------------------------------
    #  plotting

    # -------------------------------------------------------------------------
    #  style affairs

    N = len(variable) if not n_obs else min(len(variable), int(n_obs))

    # get  color, s, alhpa  from  data  if they are proper column names

    if isinstance(alpha, str):
        alpha = udf.get_column_or_except(alpha, data, "(on `alpha` parameter)")

    if isinstance(s, str):
        s = udf.get_column_or_except(s, data, "(on `s` parameter)")

    # take color from data only if it's not a color name
    if isinstance(color, str) and not h.is_mpl_color(color):
        color = udf.get_column_or_except(color, data, "(on `color` parameter)")

    color_data: pd.Series | str | None = color
    if not isinstance(color, str) and isinstance(color, Iterable):
        color = None

    style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha = \
        h.style_affairs(style, color, grid, axescolor, suptitlecolor, titlecolor, brightness, alpha, N)

    if color_data is None:
        color_data = color   # `color_data` is now valid color name or series of colors from `data`

    # -----------------------------------------------------
    # sampling and aligning color, size, alpha (if they are series)

    def sample_and_align(
            variable, covariate, n_obs=n_obs, shuffle=shuffle, random_state=random_state,
            color=color, s=s, alpha=alpha, color_data=color_data
    ):
        df = pd.concat([variable, covariate], axis=1)
        df, color, s, alpha, color_data = udf.align_sample(
            df, n_obs, shuffle, random_state,
            color=color, s=s, alpha=alpha, color_data=color_data
        )
        df, color, s, alpha, color_data = udf.align_nonas(
            df,
            color=color, s=s, alpha=alpha, color_data=color_data
        )
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
        # # ---------

        # definitions for the axes
        left, width = 0.1, 0.7
        bottom, height = 0.1, 0.7
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.145]
        rect_histy = [left + width + spacing, bottom, 0.149, height]

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

        # ---------
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
        # ---------

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
        # ---------
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
        # ---------
        h.set_xscale(ax, scale)
        h.set_grid(ax, off="both", grid=grid)
        #
        return dict(ax=ax, result=result)

    def distr(ax, variable, factor, cats, cat_colors, cmap, title="distributions by levels", legend=legend):
        """
        """
        h.set_title(ax, title, titlecolor)
        # ---------
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

    def boxplots(ax, variable, factor, cats, cat_colors, cmap, title="box-plots", horizontal=True, color=color):
        """
        future:
            - violinplots
        """
        h.set_title(ax, title, titlecolor)
        # ---------

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

        # ---------
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

        h.set_xscale(resc['axes']['ax'], xscale or scale)
        h.set_xscale(resc['axes']['ax_histx'], xscale or scale)

        h.set_yscale(resc['axes']['ax'], yscale or scale)
        h.set_yscale(resc['axes']['ax_histy'], yscale or scale)

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

        # ---------------------------------------
        # for potentially processed data    -- but NOT sampled yet (i.e. always all data!)
        df1 = pd.concat([num, fac], axis=1)
        df1agg = df1.groupby([fac.name]).agg([len, np.mean])
        df1agg = df1agg.droplevel(level=0, axis=1).sort_values(by=["len"], ascending=False)

        if data_were_processed:
            df0 = df0.loc[df.index, :]
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

        for t in ["grouped_cloud", "densities", "distr"]:
            if t in what:
                ax = axs[np.nonzero(what == t)][0]
                try:
                    result['plot'][t] = PLOTS[t]["plot"](ax, num, fac, cats, cat_colors, cmap, PLOTS[t]["name"])
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
