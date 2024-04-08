# -*- coding: utf-8 -*-
#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: NumPy And Pandas utilities
subtitle:
version: 1.0
type: code
keywords: [pd.Series, pd.DataFrame, numpy, pandas]
description: |
remarks:
todo:
sources:
file:
    date: 2019-11-20
    authors:
        - nick: arek
"""

#%%
import numpy as np
import pandas as pd
from typing import Union

from utils.builtin import lengthen, paste


#%%
def which(x: Union[np.array, pd.Series], val: Union[str, float, int]) -> int:
    """
    which(np.array(['a', 'a', 'b', 'c', 'b', 'a']), 'a')    # array([0, 1, 5])
    """
    return np.arange(len(x))[x == val]


#%%
def probability(x: np.ndarray) -> np.ndarray:
    """
    probability(np.array([1, 2, 3, 1]))     # array([0.14285714, 0.28571429, 0.42857143, 0.14285714])
    """
    return abs(x)/np.nansum(abs(x))


p = probability
prob = probability


def center(x: np.ndarray) -> np.ndarray:
    """
    center(np.array([0, np.nan, 1, np.nan, 3, np.nan, 4]))      # array([-2., nan, -1., nan,  1., nan,  2.])
    """
    return x - np.nanmean(x)


#%%
def data_frame_0(letters, numbers, index=False, columns=None) -> pd.DataFrame:
    # depricated
    df = pd.DataFrame(paste(letters, numbers, False, flat=False))

    if isinstance(index, bool):
        if index:
            index = numbers
        else:
            index = range(len(numbers))
    df.index = index

    if columns is None:
        df.columns = letters
    else:
        df.columns = columns

    return df


#%%
def data_frame(letters, numbers, index=False, columns=None) -> pd.DataFrame:
    """
    letters: list of letters;
        (may be numbers too but this is not good for names to begin with number)
    numbers: list of lists of numbers;
        `numbers` should have the same length as `letters`;
        each of sublist of `numbers` must have equal length;
        i.e. `numbers` is rectangular;

    data_frame(list("ABC"), [range(3), range(3,6), range(6,9)])
    #     A   B   C
    # 0  A0  B3  C6
    # 1  A1  B4  C7
    # 2  A2  B5  C8
    """
    if not isinstance(numbers[0], (range, list, tuple, set, dict)):
        numbers = [numbers]

    for k in range(1, len(numbers)):
        if len(numbers[k]) != len(numbers[0]):
            raise ValueError('All sublists of `numbers` must have equal length.')

    if len(numbers) != len(letters):
        numbers = lengthen(numbers, len(letters))

    dic = dict(map(lambda l, nums: (l, paste(l, nums)), letters, numbers))
    df = pd.DataFrame(dic)

    if isinstance(index, bool):
        if index:
            index = numbers[0]
        else:
            index = range(len(numbers[0]))
    df.index = index

    if columns is None:
        df.columns = letters
    else:
        df.columns = columns

    return df



#%%
"""
import re

df = tfidf_df.to_string()
df

df1 = re.sub("0(.0+ )*", ". ", df)
print(df1)
"""


#%%
def print_df(df: pd.DataFrame, zero_fill: str="."):
    """
    V 0.1    !!!
    how to deal with 0 and NaNs in one data frame ???
    """
    if not zero_fill is None:
        df[df==0] = np.nan
        dfs =  df.to_string(na_rep=zero_fill)
    print(dfs)

#%%
