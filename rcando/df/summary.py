#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Info and summary on pd.DataFrame
version: 1.0
type: module
keywords: [data frame, info/summary table]
description: |
content:
    -
remarks:
todo:
    - use try/except to serve more types; how to serve exceptions? stop! (~112)
    - round safely (i.e. only to significant digits !!!) (~331)
sources:
file:
    usage:
        interactive: False
        terminal: False
    date: 2021-10-30
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arek@staart.pl
"""

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from scipy.stats import entropy
# from common.builtin import flatten, coalesce


# %%
def print0(df: pd.DataFrame, zero_fill: str = None, na_rep: str = "-", *args, **kwargs):
    """
    V 0.1    !!!
    how to deal with 0 and NaNs in one data frame ???
    It's difficult as there is no relevant option in pd.DataFrame.to_string().
    Moreover 0 may be displayed with variable decimal places
    what makes it hard to properly replace within a string repr of data frame.

    The solution here is provisonal:
    we assume that there are no NaNs when we replace 0s
    - this way we may replace 0s with NaNs and then with anything passed
    to `na_rep` in pd.DataFrame.to_string().
    """
    if zero_fill is not None:
        df[df == 0] = np.nan
        dfs = df.to_string(na_rep=zero_fill, *args, **kwargs)
    elif na_rep is not None:
        dfs = df.to_string(na_rep=na_rep, *args, **kwargs)
    else:
        dfs = df.to_string(*args, **kwargs)

    print(dfs)


# %%
def info(df: pd.DataFrame,
         what=("dtype", "oks", "oks_ratio", "nans_ratio", "nans", "uniques", "most_common", "most_common_ratio"),
         add=None,  # "position", "most_common_value", "negatives", "zeros", "positives",
                    # "mean", "median", "min", "max", "range", "dispersion", "iqr"
         sub=None,  # things (stats) we do not want (even if otherwise we want "all"/"everything")
         #
         omit=None,     # columns/variables to omit
         dtypes=None,   # only these types; if None then all types
         exclude=None,  # but not these types
         round=None,
         name_as_index=True,
         short_names=False,
         exceptions=False
         ) -> pd.DataFrame:
    """
    Basic information on columns of data frame `df`.
     Remarks:
    - 'most_common_ratio' is the ratio of the most common value to all no-NaN values
    - 'position' is the position of the column/variable in a `df`
    All posible items (statistics/infos) are:
        dtype
        position
        oks
        oks_ratio
        nans_ratio
        nans
        uniques
        most_common
        most_common_ratio
        most_common_value
        negatives
        zeros
        positives
        mean
        median
        min
        max
        range
        dispersion
        iqr
    """
    ALL = ["dtype", "position", "oks", "oks_ratio", "nans_ratio", "nans",
           "uniques", "most_common", "most_common_ratio", "most_common_value",
           "negatives", "zeros", "positives",
           "mean", "median", "min", "max", "range", "dispersion", "iqr"]

    # !!! use try/except to serve more types; how to serve exceptions? stop!
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    if isinstance(what, str):
        what = [what]
    what = list(what)

    if add:
        if isinstance(add, str):
            add = [add]
        what.extend(add)

    if sub:
        if isinstance(sub, str):
            sub = [sub]
    else:
        sub = []

    if len(set(["all", "everything"]).intersection(what)) > 0:
        what.extend([w for w in ALL if w not in what])
        # add everything from ALL but avoid repetition
        # and preserve as much order as possible
        sub.extend(["all", "everything"])

    if len(sub) > 0:
        what = [w for w in what if w not in sub]

    if omit and isinstance(omit, str):
        omit = [omit]

    NUMS = ["float", "float64", "int", "int64", "int32", "int16",
            "int8", "uint64", "uint32", "uint16", "uint8"]

    DATES = ["datetime64[ns]", "datetime64"]

    if dtypes is not None:
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        if "numeric" in dtypes:
            dtypes.extend(NUMS)
        if len(set(["date", "time", "datetime", "datetime64[ns]", "datetime64"]).intersection(dtypes)) > 0:
            dtypes.extend(DATES)

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        if "numeric" in exclude:
            exclude.extend(NUMS)
        if len(set(["date", "time", "datetime", "datetime64[ns]", "datetime64"]).intersection(exclude)) > 0:
            exclude.extend(DATES)

    N = df.shape[0]
    info = pd.DataFrame()

    OKs = None
    NANs = None
    ss_vc = None      # = ss.value_counts()
    ss_min = None
    ss_max = None

    col_position = 0

    #  ------------------------------------------------
    #  core: NaNs & OKs

    def dtype(ss):
        return ss.dtype.__str__()   # !!!

    def oks(ss):
        return OKs

    def nans(ss):
        return NANs

    def oks_ratio(ss):
        return OKs / N

    def nans_ratio(ss):
        return NANs / N

    #  ------------------------------------------------
    #  statistics

    def negatives(ss):
        return sum(ss < 0)

    def zeros(ss):
        return sum(ss == 0)

    def positives(ss):
        return sum(ss > 0)

    def uniques(ss):
        return len(ss_vc)

    def most_common(ss):
        return ss_vc.max()

    def most_common_ratio(ss):
        return ss_vc.max() / OKs

    def most_common_value(ss):
        return ss_vc.index[0]

    def mean(ss):
        return ss.mean()

    def median(ss):
        return ss.median()

    def min_(ss):
        return ss_min

    def max_(ss):
        # nonlocal ss_max
        return ss_max

    def rng(ss):
        return ss_max - ss_min

    def dispersion(ss):
        if ss.dtype in NUMS:
            var = np.sqrt(ss.var())
        else:
            # "relative" entropy (wrt. maximum entropy -- NOT a KL-distance)
            # the smaller value the less variability i.e. ~ "near-zero-variance"
            var = 0. if len(ss_vc) == 1 else entropy(ss_vc / OKs) / entropy(np.ones(len(ss_vc)) / len(ss_vc))
        return var

    def iqr(ss):
        return ss.quantile(.75) - ss.quantile(.25)

    INFOS = {
        "dtype": {"name": "dtype", "abbr": "dtype", "stat": dtype},
        "position": {"name": "position", "abbr": "pos", "stat": lambda ss: col_position},
        "oks": {"name": "OKs", "abbr": "OKs", "stat": oks},
        "nans": {"name": "NaNs", "abbr": "NaNs", "stat": nans},
        "oks_ratio": {"name": "OKs_ratio", "abbr": "OKs2all", "stat": oks_ratio},
        "nans_ratio": {"name": "NaNs_ratio", "abbr": "NaNs2all", "stat": nans_ratio},
        "uniques": {"name": "uniques_nr", "abbr": "uniq", "stat": uniques},
        "most_common": {"name": "most_common", "abbr": "mc", "stat": most_common},
        "most_common_ratio": {"name": "most_common_ratio", "abbr": "mc2all", "stat": most_common_ratio},
        "most_common_value": {"name": "most_common_value", "abbr": "mcv", "stat": most_common_value},
        "negatives": {"name": "<0", "abbr": "<0", "stat": negatives},
        "zeros": {"name": "=0", "abbr": "=0", "stat": zeros},
        "positives": {"name": ">0", "abbr": ">0", "stat": positives},
        "mean": {"name": "mean", "abbr": "mean", "stat": mean},
        "median": {"name": "median", "abbr": "median", "stat": median},
        "min": {"name": "min", "abbr": "min", "stat": min_},
        "max": {"name": "max", "abbr": "max", "stat": max_},
        "range": {"name": "range", "abbr": "range", "stat": rng},
        "iqr": {"name": "IQR", "abbr": "IQR", "stat": iqr},
        "dispersion": {"name": "dispersion", "abbr": "disp", "stat": dispersion}, }

    name = "abbr" if short_names else "name"

    #  ----------------------------------------------------

    for n, ss in df.iteritems():

        col_position += 1

        if (dtypes and not dtype(ss) in dtypes) or \
           (exclude and dtype(ss) in exclude) or (omit and n in omit):
            continue

        item = {"name": [n]}

        ss = ss.dropna()

        if len(set(["oks", "nans", "oks_ratio", "nans_ratio", "dispersion", "most_common_ratio"])
               .intersection(what)) > 0:
            OKs = len(ss)       # !!!
            NANs = N - OKs

        if len(set(["uniques", "most_common", "most_common_ratio", "most_common_value", "dispersion"]).
               intersection(what)) > 0:
            ss_vc = ss.value_counts()

        if len(set(["min", "max", "range"]).intersection(what)) > 0:
            try:
                ss_min, ss_max = ss.min(), ss.max()
            except Exception:
                ss_min, ss_max = None, None

        for w in what:
            if w in INFOS.keys():
                try:
                    item[INFOS[w][name]] = [INFOS[w]['stat'](ss)]
                except Exception as e:
                    if exceptions:
                        print(e)
                    item[INFOS[w][name]] = [None]
            else:
                try:
                    item[w.__name__] = [w(ss)]
                except Exception as e:
                    if exceptions:
                        print(e)
                    item[str(w)] = [None]

        info = pd.concat([info, pd.DataFrame(item)])

        OKs = None
        NANs = None
        ss_vc = None
        ss_min = None
        ss_max = None

        #  END OF LOOP

    info.reset_index(inplace=True, drop=True)

    if name_as_index:
        info.index = info['name']
        info.drop(columns=['name'], inplace=True)

    # !!! round safely (i.e. only to significant digits !!!)
    if round:
        info = info.round(round)

    return info


# %%
def summary(
        df: pd.DataFrame,
        what=("dtype", "negatives", "zeros", "positives", "mean", "median", "min", "max", "range", "iqr", "dispersion"),
        add=None,  # ("pos", "oks", "oks_ratio", "nans_ratio", "nans", "uniques",
                   #  "most_common", "most_common_ratio", "most_common_value"),
        sub=None,
        #
        omit=None,
        dtypes=None,
        exclude=None,
        round=None,
        name_as_index=True,
        short_names=False,
        exceptions=False) -> pd.DataFrame:
    """
    version of info() focused by default on statistical properties (mean, meadian, etc)
    rather then technical (like NaNs number and proportions);
    i.e. info() with different (kind of reversed) defaults
    """
    summary = info(df, what=what, add=add, sub=sub,
                   omit=omit, dtypes=dtypes, exclude=exclude,
                   round=round, exceptions=exceptions,
                   short_names=short_names)

    return summary

# %%
