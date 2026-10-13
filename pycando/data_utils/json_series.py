#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: json (of any depth) or series of jsons into data frame
version: 1.0
type: module             # module, analysis, model, tutorial, help, example, ...
keywords: [json, data frame, series of jsons]
description: |
remarks:
todo:
    - json series to list of data frames;
file:
    date: 2022-12-02
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import gc
from typing import Iterable, Tuple, Any, Dict
import json
import pandas as pd
import utils.builtin as bi


# %%
@bi.timeit
def df_from_json_series(
    jss: pd.Series,
    batch: int = int(1e4),
    replace: Tuple[Iterable[Any], Any] = (["", "none", "None", "NaN", "null", "Null", "NULL"], None),
    json_normalize: Dict = dict(),
    print_batch: bool = True
) -> pd.DataFrame:
    """
    Turning pd.Series of json strings `jss` into data frame having the same length
    as `jss` and columns with exactly 2 levels.
    This is faster version of a solution to the task and have nicer output
    -- two level columns are easier to manage later
    (compare df_from_json_series_0()).

    It's STRONGLY ASSUMED that:
    - jss is pd.Series of jsons of the same structure
    - and having depth at least 2, e.g. {'a': {'p': v, ...}, ..., 'b': {'q': ...}}.
    - each leaf of a json is of simple type (float, int, str) not a list.

    Output is data frame with 2-level columns.
    Index is inherited from `jss`.

    It is possible to replace some values with other via `replace` where
    arguments to common.builtin.replace() are passed.

    Data are read in `batch`es as very large input crashes kernel (memory issues).

    The main engine in turning jsons string into data frame is pandas.json_normalize()
    to which parameters may be passed via dictionary `json_normalize`.
    """
    res = pd.DataFrame()
    batch = int(batch)
    nb = len(jss) // batch + 1
    for b in range(nb):
        if print_batch:
            print(b * batch)
        idx = slice(b * batch, (b + 1) * batch)
        idx = jss[idx].index    # because of (*)
        if len(idx) == 0:       # happens when  batch  divides  len(ss)
            break
        dfb = jss[idx].apply(json.loads)
        dfb = dfb.apply(bi.replace_deep, args=replace)
        # core ------
        upper_keys = list(dfb.iloc[0].keys())
        dfb = [(v for k, v in js.items()) for js in dfb.to_list()]
        dfb = {k: pd.Series(list(s)) for k, s in zip(upper_keys, zip(*dfb))}
        dfb = {k: pd.json_normalize(s, **json_normalize) for k, s in dfb.items()}
        dfb = pd.concat(dfb, axis=1)
        # -----------
        dfb.index = idx         # (*)
        res = pd.concat([res, dfb])
        gc.collect()    # not very helpfull but safe
    return res


# %%
@bi.timeit
def df_from_json_series_0(
    jss: pd.Series,
    batch: int = int(1e4),
    replace: Tuple[Iterable[Any], Any] = (["", "none", "None", "NaN", "null", "Null", "NULL"], None),
    json_normalize: Dict = dict(),
    print_batch: bool = True
) -> pd.DataFrame:
    """
    Turning pd.Series of json strings `jss` into data frame having the same length
    as `jss` and columns having only one level with names composed of
    the whole path to data.
    It means the column names are rather ugly if jsons are deep.
    This is first version of a solution to the task
    and it's both way over 10% slower and result is bit ugly
    -- one level columns are hard to read and not convenient for later management
    (see df_from_json_series() which is faster and prettier).

    It's strongly assumed that each leaf of a json is a simple type (float, int, str) not a list.
    Output is data frame with 1-level columns.
    Index is inherited from `jss`.

    It is possible to replace some values with other via `replace` where
    arguments to common.builtin.replace() are passed.

    Data are read in `batch`es as very large input crashes kernel (memory issues).

    The main engine in turning jsons string into data frame is pandas.json_normalize()
    to which parameters may be passed via dictionary `json_normalize`.
    """
    res = pd.DataFrame()
    batch = int(batch)
    nb = len(jss) // batch + 1
    for b in range(nb):
        if print_batch:
            print(b * batch)
        idx = slice(b * batch, (b + 1) * batch)
        idx = jss[idx].index    # because of (*)
        if len(idx) == 0:       # happens when  batch  divides  len(ss)
            break
        dfb = jss[idx].apply(json.loads)
        dfb = dfb.apply(bi.replace_deep, args=replace)
        # core ------
        dfb = pd.json_normalize(dfb, **json_normalize)
        # -----------
        dfb.index = idx         # (*)
        res = pd.concat([res, dfb])
        gc.collect()    # not very helpfull but safe
    return res


# %%
