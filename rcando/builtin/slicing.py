#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

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
sources:
file:
    usage:
        interactive: True
        terminal: False
    date: 2023-01-16
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

# %%
import numpy as np
import pandas as pd


# %%
def boolean_slice(n: int, slic: slice) -> list[bool]:
    """
    useful only with np.arrays or pd.Series / pd.DataFrame
    (used for some trials)
    """
    seqn = list(range(n))
    seq_true = seqn[slic]
    seq_bool = [i in seq_true for i in seqn]
    return seq_bool


def parse_slice(slic: [str, slice]) -> (bool, slice):
    """
    parse_slice('1:')     # (False, slice(1, None, None))
    parse_slice(':2')     # (False, slice(None, 2, None))
    parse_slice('~:-1')   # (True, slice(None, -1, None))
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
    return slic, neg


def subseq(seq: list, slic: str) -> list:
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
    """
    slic, neg = parse_slice(slic)
    sseq = seq[slic]
    if neg:
        sseq = [s for s in seq if s not in sseq]
    return sseq


def subseq_np(seq: np.ndarray, slic: str) -> np.ndarray:
    """
    ll = np.arrange(10)
    subseq_np(ll, '1:')      # array([1, 2, ..., 9])
    # ... as for subseq()
    """
    slic, neg = parse_slice(slic)
    sseq = seq[slic]
    if neg:
        sseq = seq[~np.isin(seq, sseq)]
    return sseq


def subseq_ss(seq: pd.Series, slic: str, loc=False, if_empty: str = "none") -> pd.Series:
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
    #  ...
    """
    slic, neg = parse_slice(slic)
    if loc:
        sseq = seq.loc[slic]
    else:   # default
        sseq = seq.iloc[slic]
    if neg:
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


def subseq_df(df: pd.DataFrame, slic: str, loc=False, if_empty: str = "none") -> pd.DataFrame:
    """
    as subseq_ss() but for data frames
    """
    slic, neg = parse_slice(slic)
    if loc:
        sdf = df.loc[slic, :]
    else:   # default
        sdf = df.iloc[slic, :]
    if neg:
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
