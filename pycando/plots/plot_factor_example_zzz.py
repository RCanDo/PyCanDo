#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:31:57 2023

@author: arek
"""

# %%
import numpy as np
import pandas as pd
from utils.plots.plot_factor import plot_factor
from utils.plots import plot_variable
from utils.config import pandas_options
pandas_options()

# %%
N = 222

a = pd.Series(np.random.choice(list('abcd'), N, True, (.4, .3, .2, .1)))
none_idx = np.random.choice(N, size=(22,), replace=False)
a.iloc[none_idx] = None
a

# %%
avc = a.value_counts(dropna=False)
avc
avc.index
type(avc.index[-1])     # NoneType

plot_factor(a)
plot_factor(a, dropna=True)

plot_variable(a.astype(str))

# %%
b = pd.Series(np.random.randint(11, size=(N,)))
none_idx = np.random.choice(N, size=(33,), replace=False)
b.iloc[none_idx] = None
b   # there are NaN not None's !
b.dtype     # float64

type(b[none_idx[0]])    # numpy.float64

# %%
bvc = b.value_counts(dropna=False)
bvc
bvc.index
bvc.index[0]            # nan
type(bvc.index[0])      # numpy.float64

plot_factor(b)
plot_factor(b, dropna=True)

plot_variable(b)
plot_variable(b, dropna=True)

# %%
