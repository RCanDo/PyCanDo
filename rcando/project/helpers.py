#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:03:01 2022

@author: arek
"""

import pathlib
import pandas as pd
from typing import Iterable
pd.Timestamp

#%%
def to_dict(x):
    """
    generality for transforming in dictionary
    without
    """
    if isinstance(x, pathlib.PosixPath):
        res = str(x)
    elif isinstance(x, str):
        res = x
    #elif isinstance(x, pd.Series):
    #    res = x.to_list()
    elif isinstance(x, Iterable):
        try:
            res = {k: to_dict(v) for k, v in x.items()}
        except:
            res = [to_dict(v) for v in x]
    else:
        try:
            res = float(x)
        except:
            res = str(x)

    return res

#%%
"""
isinstance(pd.Timestamp('2021-11-30 00:00:00'), pd.Timestamp)
str(pd.Timestamp('2021-11-30 00:00:00'))
float(pd.Timestamp('2021-11-30 00:00:00'))
"""
