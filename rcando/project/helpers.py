#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:03:01 2022

@author: arek
"""

# %%
import pathlib
from typing import Iterable
import pandas as pd


# %%
def to_dict(x, series_as_list=True):
    """
    crude generality for transforming to dictionary;
    retains lists so it works almost like transforming to json;

    Examples
    --------

    obj = {'a': [{"x": 2, "y":["a", {11: 'aa', 22:"bb"}, pd.DataFrame({'a':[1, 2], 'b':[33, 22]})]}, 2],
           "b": {'p': (3, 4), "g": 2},
           "c": pd.DataFrame({'a':[1, 2], 'b':[33, 22]}),
           "d": pd.Series([1, 2, 3, 4, 5])
           }
    obj
    to_dict(obj)
    to_dict(obj, False)

    obj = {'a': [2, 3, 2], "b": {'p': (3, 4), "g": 2}, "c": None, "d": pd.Timestamp('2022-02-22 22:22')}
    to_dict(obj)
    """
    if isinstance(x, pathlib.PosixPath):
        res = str(x)
    elif isinstance(x, str):
        res = x
    elif isinstance(x, pd.Series) and series_as_list:
        res = x.to_list()
    elif isinstance(x, Iterable):
        try:
            res = {k: to_dict(v, series_as_list) for k, v in x.items()}
        except Exception:
            res = [to_dict(v, series_as_list) for v in x]
    else:
        try:
            res = float(x)
        except Exception:
            res = str(x)

    return res


# %%
