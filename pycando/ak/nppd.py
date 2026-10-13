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
from numpy.typing import NDArray
import pandas as pd

from ..builtin import lengthen, paste  # type: ignore  [import-not-found]
from ..builtin import align as bi_align

from collections.abc import Sequence
from typing import Union, Tuple, TypeVar, Any
from ..mytypes import ValuesStrict, NpInt  # type: ignore  [import-not-found]

T = TypeVar("T", bound=Any)


#%%
def which(
        x: Union[Sequence[T], NDArray[T], "pd.Series[T]"],
        val: Union[T, Sequence[T], NDArray[T], "pd.Series[T]"],
) -> Tuple[NDArray[NpInt], ...]:
    """
    Shorthand for `np.where(np.isin(x, val))`.

    which(['a', 'a', 'b', 'c', 'b', 'a'], 'a')                # (array([0, 1, 5]),)
    which(np.array(['a', 'a', 'b', 'c', 'b', 'a']), 'a')      # (array([0, 1, 5]),)
    which(pd.Series(['a', 'a', 'b', 'c', 'b', 'a']), 'a')     # (array([0, 1, 5]),)
    # find in 2d array
    arr = np.array([['a', 'a', 'b'],
                    ['c', 'b', 'a'],
                    ['b', 'a', 'c']])
    which(arr, 'a')  # (array([0, 0, 1, 2]),   1st dim – rows
                     #  array([0, 1, 2, 1]))   2nd dim – cols

    # find many values
    which(['a', 'a', 'b', 'c', 'b', 'a'], ['a', 'c'])         # (array([0, 1, 3, 5]),)
    """
    return np.where(np.isin(x, val))


def probability(x: ValuesStrict) -> ValuesStrict:
    """
    Shorthand for `abs(x)/np.nansum(abs(x))`.

    probability(np.array([1, 2, 3, 1]))     # array([0.14285714, 0.28571429, 0.42857143, 0.14285714])
    """
    return abs(x)/np.nansum(abs(x))


p = probability
prob = probability


def center(x: ValuesStrict) -> ValuesStrict:
    """
    Shorthand for `x - np.nanmean(x)`.

    center(np.array([0, np.nan, 1, np.nan, 3, np.nan, 4]))   # array([-2., nan, -1., nan,  1., nan,  2.])
    """
    return x - np.nanmean(x)


#%%
def data_frame(
        columns: list[str|int],
        rows: list[list[str|int]],
        index: bool | list = False,
        colnames: None | list = None,
        align: str = 'lengthen',
) -> pd.DataFrame:
    """
    Example data frame for testing of the form (example):
         A   B   C
     0  A0  B3  C6
     1  A1  B4  C7
     2  A2  B5  C8
    where ['A', 'B', 'C'] are passed to `columns` and `[(0, 1, 2), (3, 4, 5), (6, 7, 8)]` are passed to `rows`.
    Both of `columns` and `rows` may be either `str` or `int`
    (however it is not good for column names to begin with number).
    Moreover, it is also possible to set particular index values or column names
    independently of `columns` and `rows` .

    Arguments
    ---------
    columns: list[str|int],
        The prefix of each value for each column separately;
        also default name of each column if column names not passed explicitely to `colnames`.
    rows: list[list[int|str]],
        `rows` should have the same length as `columns`;
        each element of `rows` is a sequence of suffixes for values in respective column.
        all elements of `rows` should have equal length, i.e. `rows` should be rectangular,
        however, if they are not equal then are aligned according to `align` strategy;
        `len(<longest rows>)` is the number of rows in the final data frame when `align` is "lengthen" (default);
    index: bool | list = False,
        if False (default) ordinary positional index is created;
        if True index is made of `rows[0]`;
        otherwise pass the idex values directly (within list or any Sequence);
    colnames: None | list = None,
        if None then column names are created from `columns`
        otherwise pass the column names directly (within list or any Sequence);
    align: str = 'lengthen',
        strategy for aligning sequences in `rows`, one of 'lengthen', 'prod', 'cycle', 'lcm'.

    Examples
    --------
    data_frame(list("ABC"), [range(3), range(3,6), range(6,9)])
    #     A   B   C
    # 0  A0  B3  C6
    # 1  A1  B4  C7
    # 2  A2  B5  C8
    data_frame(list("ABC"), range(4))
    #     A   B   C
    # 0  A0  B0  C0
    # 1  A1  B1  C1
    # 2  A2  B2  C2
    # 3  A3  B3  C3

    # uneven `rows`
    data_frame(list("ABC"), [range(3), range(2)])
    #     A   B   C
    # 0  A0  B0  C0
    # 1  A1  B1  C1
    # 2  A2  B1  C1
    data_frame(list("ABC"), [range(3), range(2), range(4)])
    #     A   B   C
    # 0  A0  B0  C0
    # 1  A1  B1  C1
    # 2  A2  B1  C2
    # 3  A2  B1  C3

    data_frame(list("ABC"), [range(3), range(2), range(4)], align='cycle')
    # 0  A0  B0  C0
    # 1  A1  B1  C1
    # 2  A2  B0  C2
    # 3  A0  B1  C3
    data_frame(list("ABC"), [range(3), range(2), range(4)], align='lcm')
    # 12 rows = least common multiplicity of 3, 2, 4 – indices for each column fully recycled:
    #      A   B   C
    # 0   A0  B0  C0
    # 1   A1  B1  C1
    # ...
    # 10  A1  B0  C2
    # 11  A2  B1  C3
    data_frame(list("ABC"), [range(3), range(2), range(4)], align='prod')
    # all possible combinations of numbers for each column – 24 rows

    data_frame(list("ABC"), range(2, 5))
    #     A   B   C
    # 0  A2  B2  C2
    # 1  A3  B3  C3
    # 2  A4  B4  C4
    data_frame(list("ABC"), range(2, 5), index=True)
    #     A   B   C
    # 2  A2  B2  C2
    # 3  A3  B3  C3
    # 4  A4  B4  C4
    data_frame(list("ABC"), range(2, 5), index=list('pqr'))
    #     A   B   C
    # p  A2  B2  C2
    # q  A3  B3  C3
    # r  A4  B4  C4
    data_frame(list("ABC"), list('pqr'), index=range(2, 5))
    #     A   B   C
    # 2  Ap  Bp  Cp
    # 3  Aq  Bq  Cq
    # 4  Ar  Br  Cr
    data_frame(list("abc"), range(2, 5))
    #     a   b   c
    # 0  a2  b2  c2
    # 1  a3  b3  c3
    # 2  a4  b4  c4
    data_frame(list("abc"), range(2, 5), colnames=list('ABC'))
    #     A   B   C
    # 0  a2  b2  c2
    # 1  a3  b3  c3
    # 2  a4  b4  c4
    data_frame(range(3), range(2, 5))
    #     0   1   2
    # 0  02  12  22
    # 1  03  13  23
    # 2  04  14  24     these are strings!
    data_frame(range(3), range(2, 5), colnames=list('PQR')).astype(int)
    #    P   Q   R
    # 0  2  12  22
    # 1  3  13  23
    # 2  4  14  24
    """
    if not isinstance(rows[0], (range, list, tuple, set, dict)):
        rows = [rows]

    rows = bi_align(*rows, mode=align)

    if len(rows) != len(columns):
        rows = lengthen(rows, len(columns))

    dic = dict(map(lambda c, rr: (c, paste(c, rr)), columns, rows))
    df = pd.DataFrame(dic)

    if isinstance(index, bool):
        if index:
            index = rows[0]
        else:
            index = range(len(rows[0]))
    df.index = index

    if colnames is None:
        df.columns = columns
    else:
        df.columns = colnames

    return df


#%%
def print_df(df: pd.DataFrame, zero_fill: str = ".") -> None:
    """
    V 0.1    !!!
    how to deal with 0 and NaNs in one data frame ???

    df = pd.DataFrame(
        np.random.randint(3, size=12).reshape(3,4),
        columns=list("abcd")
        )
    print_df(df)
    """
    if not zero_fill is None:
        df[df==0] = np.nan
        dfs = df.to_string(na_rep=zero_fill)
    print(dfs)


"""
import re

df = tfidf_df.to_string()
df

df1 = re.sub("0(.0+ )*", ". ", df)
print(df1)
"""

# %%
