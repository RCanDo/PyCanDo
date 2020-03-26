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
    Convieniens functions an utilities used in everyday work.
content:
    - data_frame(letters, numbers, columns=None) -> pd.DataFrame
remarks:
todo:
sources:
file:
    usage:
        interactive: True
        terminal: True
    name: nppd.py
    path: D:/ROBOCZY/Python/RCanDo/ak/
    date: 2019-11-20
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

#%%
"""
pwd
cd D:/ROBOCZY/Python/RCanDo/...
ls
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional, Union, NewType

from .builtin import lengthen, flatten, paste

#%%
def which(x: Union[np.array, pd.Series], val: Union[str, float, int]):
    return np.arange(len(x))[x == val]

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

