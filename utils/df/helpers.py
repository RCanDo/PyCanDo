#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Helper functions for pd.DataFrame
version: 1.0
type: module
keywords: [data frame, align, NaN, count levels factors, datetime, ]
description: |
    Aligning series and data frames safely passing other types.
    Converting to proper datetime format.
    Readable memory usage for data frame.
content:
remarks:
todo:
sources:
file:
    date: 2022-11-05
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arek@staart.pl
"""

# %%
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')

import pandas as pd


# %%
def to_datetime(ss: pd.Series, unit: str = 's', floor: str = 's') -> pd.Series:
    """"""
    ss = pd.to_datetime(ss.astype(int, errors='ignore'), unit=unit)
    ss = ss.dt.floor(floor)
    return ss


# %%
def memory(
    df: Union[pd.DataFrame, pd.Series],
    r: int = 1,  # round
    unit: str = "MB",  # "KB", "B", "GB"
    as_number: bool = False,  #
    memory_usage: dict = dict(index=True, deep=True)
) -> Union[str, float]:
    """"""
    df = pd.DataFrame(df)  # for Series; safe, idempotent
    mem = df.memory_usage(**memory_usage).sum()
    unit_power = {"B": 0, "KB": 1, "MB": 2, "GB": 3}
    mem = round(mem / 1024**unit_power[unit], r)
    if not as_number:
        mem = f"{mem} {unit}"
    return mem


# %%
def count_factors_levels(
    df: pd.DataFrame,
    flatten: str = " : ",
    row_name: int = 0,
) -> pd.DataFrame:
    """
    Returns data frame of counts of each level of each factor of `df` .

    It is assumed that all columns of `df` are categorical (factors).

    Result is one row data frame whith columns being 2-level MultiIndex
    where the first level gives a name of a variable
    and the second level gives all levels of the respective variable (factor!).
    The only row has the name `row_name`.

    If `flatten` is not None
    then column names will be flattened to 1 level according to:
        "factor_name" + "flatten" + "factor_level"
    e.g. "variable : value" if `flatten = " : "` (default).
    """
    vars_dict = dict()
    for v in df.columns:
        vc = df[v].value_counts()
        vc.name = row_name
        vars_dict[v] = pd.DataFrame(vc)

    frame = pd.concat(vars_dict).T
    frame

    frame.fillna(0, inplace=True)  # looks like spurious

    for v in frame.columns:
        frame[v] = frame[v].astype(int)

    if flatten is not None:
        frame.columns = [flatten.join(v) for v in frame.columns]

    return frame


# %%  exactly the same in plots.helpers
def sample(
    data: Union[pd.DataFrame, pd.Series],
    n: int, shuffle: bool, random_state: int
) -> Union[pd.DataFrame, pd.Series]:
    """"""
    if n and n < len(data):
        data = data.sample(int(n), ignore_index=False, random_state=random_state)
    if shuffle:
        data = data.sample(frac=1, ignore_index=True, random_state=random_state)
    else:
        data = data.sort_index()  # it is usually original order (but not for sure...)
    return data


# %%
def align_indices(data, *args) -> List[pd.Series]:
    """
    Replace indices of all pd.Series passed to `args` with index of `data`.
    It will work only if all the series have the same length.
    data: pd.Series, pd.DataFrame;
        main data (pd.Series or pd.DataFrame)
        the index of which will replace index of evety pd.Series from `args`;
    args: arguments of any types;
        all the pd.Series from `args` will have their indices replaced
        with the index of `data`;
        objects of other types are ignored, and returned intact.
     Returns
    [data, *args]
    with changes to the `args` elements as described above.
    """
    idx = data.index
    result = [data]
    for item in args:
        if item is not None and isinstance(item, pd.Series):
            item.index = idx
            result.append(item)
        else:
            result.append(item)

    return result


# %%
def align_nonas(data, **kwargs) -> List[pd.Series]:
    """
    Align (all the pd.Series passed) wrt no-NaN values, i.e.
    merging all the series from `kwargs` using their index
    and removing all records with NaN (or None, etc.);
    data: pd.Series, pd.DataFrame;
        main data (pd.Series or pd.DataFrame),
        a data to which all the other arguments from `kwargs`
        are tried to be aligned with using their indices;
    kwargs: key-value pairs
        all values being pd.Series will be processed as described above;
        values of other types are ignored, and returned intact.
     Return
    [data, *kwargs.values()]
    with changes to the `kwargs.values()` elements as described above.
    """
    length = len(data)
    df = pd.DataFrame(data)
    used_keys_dict = dict()  # recording proper pd.Series entries of kwargs
    k0 = df.shape[1]
    k = k0 - 1

    for name, ss in kwargs.items():
        if ss is not None and isinstance(ss, pd.Series):
            k += 1
            used_keys_dict[k] = name
            ss = ss.dropna()
            df = pd.merge(df, ss, left_index=True, right_index=True, how='inner')     # 'inner' is default
            if len(df) < length:
                print(f"WARNING! There were empty entries for `{name}` or it does not align with data",
                       "-- data was pruned accordingly!\n",
                      f"Only {len(df)} records left.")
                length = len(df)

    if len(used_keys_dict) > 0:
        data = df.iloc[:, :k0] if k0 > 1 else df.iloc[:, 0]     # if...  is spurious
        for k, name in used_keys_dict.items():
            kwargs[name] = df.iloc[:, k]

    result = [data, *kwargs.values()]     # !!! order of input preserved !!!

    return result


# %%
def align_sample(data, n_obs=int(1e4), shuffle=False, random_state=2,
                 **kwargs) -> List[pd.Series]:
    """
    """
    df = pd.DataFrame(data)
    df = sample(df, n_obs, shuffle, random_state)
    used_keys_dict = dict()  # recording proper pd.Series entries of kwargs
    k0 = df.shape[1]
    k = k0 - 1

    for name, ss in kwargs.items():
        if ss is not None and isinstance(ss, pd.Series):
            k += 1
            used_keys_dict[k] = name
            df = pd.merge(df, ss, left_index=True, right_index=True, how='left')     # 'inner' is default

    if len(used_keys_dict) > 0:
        data = df.iloc[:, :k0] if k0 > 1 else df.iloc[:, 0]     # if...  is spurious
        for k, name in used_keys_dict.items():
            kwargs[name] = df.iloc[:, k]

    result = [data, *kwargs.values()]     # !!! order of input preserved !!!

    return result


# %%
def divide_df(
        df: pd.DataFrame,
        by: str,
        allowed_values: list[str] = None,
        add_all_allowed: str = "",
        add_all: str = "",
        only_index: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Wrapper around `dict(tuple(df.groupby(by)))` with some additional control.
    Given data frame `df` with column `by` splits `df` into parts wrt to values of `by`.
    Returns dictionary with unique values of `by` as keys and respective `df` parts as values.

    To spare space, by default only indices of respective parts are remembered: `only_index=True`.
    Set `only_index=False` to remember whole data parts -- notice though that it's memory consuming
    as it simply replicates data (may be killer for large `df`).

    Result is limited to only `allowed_values` of `by`,
    however, if `allowed_values` is None then all values of `by` are considered.

    To the resulting dictionary one may add the whole `df` under key `add_all` (if not empty)
    and also the whole part of `df` corresponding to only `allowed_values`
    - under key `add_all_allowed` (if not empty).

    When there is no `allowed_values` in data or `allowed_values` is empty list
    then returns empty dict (if additionally `add_all_allowed` and `add_all` are empty).

    Arguments
    ---------
    df: pd.DataFrame,
    by: str,
        column name; wrt values of this column `df` will be split;
    allowed_values: list[str] = None,
        list of values of `by` which are allowed, i.e. split will be made wrt to only these values
        and other values are ignored (ommited);
        if None all values of `by` are considered;
        ! empty list will result with no split (thus it is opposite of None case).
    add_all_allowed: str = "",
        the key under which to remember part of `df` corresponding to all `allowed_values`;
        ignored if empty;
    add_all: str = "",
        the key under which to remember whole `df`;
        ignored if empty;
    only_index: bool = False,
        to remember only indices of respective parts of `df`;
    """
    divdic = {}

    if only_index:
        df = df[[by]]

    if add_all:
        divdic |= {add_all: df}

    if allowed_values is not None:
        df = df[df[by].isin(allowed_values)]    # may be empty

    if add_all_allowed:
        divdic |= {add_all_allowed: df}

    if not df.empty:
        divdic |= dict(tuple(df.groupby(by)))

    if only_index:
        divdic = {k: d.index for k, d in divdic.items()}

    return divdic

# %%
