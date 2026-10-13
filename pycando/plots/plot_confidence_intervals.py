#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Plot sequence of Confidence Intervals
version: 1.0
type: module
keywords: [plot, confidence intervals]
description: |
    Plot sequence of Confidence Interval, CI,
    where each CI is supposed to have at least the following attributes:
        - mean
        - lower (lower)
        - upper
        - alpha
content:
remarks:
todo:
    - horizontal/vertical
    - add arbitrary horizontal/vertical/diagonal lines (ax.hlines, ax.vlines, ax.axline)
    - categorical (letters) `x` (ax.set_xticklabels)
    - scale
    - print mean values on the plot
sources:
file:
    date: 2023-09-22
    authors:
"""
# %%
from typing import Sequence
from utils.stats import CI
import matplotlib.pyplot as plt


# %%
def plot_confidence_intervals(
        x: Sequence[float],
        cis: Sequence[CI],
        title: str | None = None,
        xlabel: str = 'X',
        ylabel: str = 'means',
        diag: bool = False,         # provisional
        figsize: tuple[float, float] = (10., 10.),
) -> plt.Figure:
    """
    Plots series (list) of confidence intervals `cis` (CIs) vertically  placed at respective `x`s.

    Arguments
    ---------
    x: Sequence[float],
        sequence of points (floats) at which CIs will be plot as vertical segments;
        i.e. for every i-th CI, cis[i], its middle point will be plot at point (x[i], cis[i].mean);
    cis: Sequence[CI],
        sequence of confidence intervals (objects of class CI, or at least with attributes
        .mean, .lower, .upper and .alpha);
    title: str = None,
        title of a plot;
        if None then is set to 'confidence intervals for means (confidence level = {1 - alpha})',
        where alpha is cis[0].alpha (significance level);
        i.e. we assume that all CIs in `cis` as calculated at the same significance level `alpha`.
    xlabel: str = 'X',
        label for x-axis;
    ylabel: str = 'means',
        label for y-axis;
    diag: bool = False,
        if True then adds diagonal line (provisional)
    figsize: tuple[float, float] = (10., 10.),
        (width, height) (inches) of the figure;

    Returns
    -------
    plt.Figure

    Examples
    --------
    import numpy as np
    from utils.plots import plot_confidence_intervals as plot_cis
    from utils.stats import CI

    cis = [CI_binom(s, n) for s, n in [(33, 77), (7, 22), (78, 123)]]
    plot_cis([1, 2, 3], cis)

    n = 22
    pp = [.1, .5, .7]
    ss = [sum(np.random.choice(2, n, p=(1-p, p))) for p in pp]
    cis = [CI_binom(s, n) for s in ss]
    plot_cis(pp, cis, diag=True)
    """
    means = [ci.mean for ci in cis]
    lowers = [ci.lower for ci in cis]
    uppers = [ci.upper for ci in cis]

    fig, ax = plt.subplots(figsize=figsize)

    if diag:
        ax.axline((0, 0), slope=1, linestyle='--', color='darkgrey')

    ax.scatter(x, means, s=100, marker='_', color="r")
    ax.plot((x, x), (lowers, uppers), '_-', color='orange', ms=10)

    if title is None:
        title = f'confidence intervals for means (confidence level = {1 - cis[0].alpha})'
    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig
