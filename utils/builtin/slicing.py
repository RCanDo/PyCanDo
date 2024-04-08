#! python3
# -*- coding: utf-8 -*-
"""
---
title:
subtitle:
version: 1.0
type: module
keywords: [slice, complements, ]
description: |
    Slicing and it's complements from string.
    Convenient for some configs from file like yaml or json.
remarks:
todo:
    - try to make class extending slice to be created from strings with negation
sources:
file:
    date: 2023-01-16
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

# %%
from typing import Union
import numpy as np
import pandas as pd


# %%
def parse_slice(slic: [str, slice]) -> (slice, bool):
    """
    parse_slice('1:')     # (slice(1, None, None), True)
    parse_slice(':2')     # (slice(None, 2, None), True)
    parse_slice('~:-1')   # (slice(None, -1, None), False)
    parse_slice("~1:9:3") # (slice(1, 9, 3), False)
    """
    if isinstance(slic, slice):
        neg = False
    else:
        neg = slic[0] == "~"
        if neg:
            slic = slic[1:]
        slic = slic.strip("()")
        slic = slic.split(":")
        slic = [int(i) if i else None for i in slic]
        slic = slice(*slic)
    return slic, not neg


sslice = parse_slice    # alias: "string slice"


def boolean_slice(n: int, slic: Union[str, slice], noneg: bool = True) -> list[bool]:
    """
    useful only with np.arrays or pd.Series / pd.DataFrame;
    n: int
        lenght of the sequence for which we need boolean indices
    slic: slice, str
        ...
    noneg: bool
        if False then indices are negated;
        if True (default) then indices are taken directly;
        hence, e.g.:  boolean_slice(9, slice(1, 7, 3), False) == boolean_slice(9, "~1:7:3")
        and  boolean_slice(9, "~1:7:3", False) == boolean_slice(9, "1:7:3")
    (used for some trials)

    boolean_slice(9, slice(1, 7, 3))  # [False, True, False, False, True, False, False, False, False]
    boolean_slice(9, slice(1, 7, 3), False)  # [True, False, True, True, False, True, True, True, True]
    boolean_slice(9, "~1:7:3")  # [True, False, True, True, False, True, True, True, True]

    ll = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ll.index =     [2, 7, 1, 0, 9, 6, 3, 8, 5, 4]
    ll[bslice(10, "~1:7:3")]
    ll[bslice(10, "~2:-2")]
    """
    if isinstance(slic, str):
        slic, noneg = parse_slice(slic)
    elif isinstance(slic, slice):
        pass
    else:
        raise TypeError("`slic` must be slice or its string repr like e.g. '1:4', '1:7:2', '~1:7:2'")
    seqn = list(range(n))
    seq_true = seqn[slic]
    if noneg:
        seq_bool = [i in seq_true for i in seqn]
    else:
        seq_bool = [i not in seq_true for i in seqn]
    return seq_bool


bslice = boolean_slice  # alias


def subseq(seq: list, slic: Union[str, slice]) -> list:
    """
    ll = list(range(10))
    subseq(ll, '1:')         # [1, 2, ..., 9]
    subseq(ll, ':2')         # [0, 1]
    subseq(ll, '2')          # [0, 1]
    # take all but first
    subseq(ll, '~:1')        # [1, 2, 3, ..., 9]
    subseq(ll, '~1')         # [1, 2, 3, ..., 9]
    # take all but last
    subseq(ll, '~:-1')       # [9]
    # ! take first and last
    subseq(ll, '~1:-1')      # [0, 9]
    # other
    subseq(range(12), "~1:9:3")     # [0, 2, 3, 5, 6, 8, 9, 10, 11]
    """
    slic, noneg = parse_slice(slic)
    sseq = seq[slic]
    if not noneg:
        sseq = [s for s in seq if s not in sseq]
    return sseq


def subseq_np(seq: np.ndarray, slic: Union[str, slice]) -> np.ndarray:
    """
    ll = np.arrange(10)
    subseq_np(ll, '1:')      # array([1, 2, ..., 9])
    # ... as for subseq()
    """
    slic, noneg = parse_slice(slic)
    sseq = seq[slic]
    if not noneg:
        sseq = seq[~np.isin(seq, sseq)]
    return sseq


def subseq_ss(seq: pd.Series, slic: Union[str, slice], loc=False, if_empty: str = "none") -> pd.Series:
    """
    ll = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ll.index =     [2, 7, 1, 0, 9, 6, 3, 8, 5, 4]
    subseq_ss(ll, '8:')
    # 5    8
    # 4    9
    # dtype: int64
    subseq_ss(ll, '8:', loc=True)
    # 8    7
    # 5    8
    # 4    9
    # dtype: int64
    subseq_ss(ll, '1:8:2', loc=True)
    # 1    2
    # 9    4
    # 3    6
    # dtype: int64
    subseq_ss(ll, '~1:8:2', loc=True)
    # 2    0
    # 7    1
    # 0    3
    # 6    5
    # 8    7
    # 5    8
    # 4    9
    # dtype: int64
    #  ...
    """
    slic, noneg = parse_slice(slic)
    if loc:
        sseq = seq.loc[slic]
    else:   # default
        sseq = seq.iloc[slic]
    if not noneg:
        sseq = seq.loc[(~seq.isin(sseq))]
    if len(sseq) == 0:
        if if_empty == "all":
            sseq = seq
        elif if_empty == "first":
            sseq = seq.iloc[[0]]
        elif if_empty == "last":
            sseq = seq.iloc[[-1]]
        elif if_empty == "random":
            idx = np.random.randint(len(seq), size=1)[0]
            sseq = seq.iloc[[idx]]
        elif if_empty == "none":
            pass
        else:
            pass
    return sseq


subseq_pd = subseq_ss   # alias


def subseq_df(df: pd.DataFrame, slic: Union[str, slice], loc=False, if_empty: str = "none") -> pd.DataFrame:
    """
    as subseq_ss() but for data frames
    """
    slic, noneg = parse_slice(slic)
    if loc:
        sdf = df.loc[slic, :]
    else:   # default
        sdf = df.iloc[slic, :]
    if not noneg:
        sdf = df.loc[(~df.index.isin(sdf.index)), :]
    if len(sdf) == 0:
        if if_empty == "all":
            sdf = df
        elif if_empty == "first":
            sdf = df.iloc[[0], :]
        elif if_empty == "last":
            sdf = df.iloc[[-1], :]
        elif if_empty == "random":
            idx = np.random.randint(len(df), size=1)[0]
            sdf = df.iloc[[idx], :]
        elif if_empty == "none":
            pass
        else:
            pass
    return sdf

# %%
