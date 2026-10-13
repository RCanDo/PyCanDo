#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:28:34 2023

@author: arek
"""

# %% Example
import numpy as np
import pandas as pd
from utils.plots import plot_covariates
from utils.config import pandas_options
pandas_options()

from utils.sklearn_ectansions import StratifiedGroupShuffleSplit

# %%
N = 1000
y = pd.Series(np.random.choice(2, N, True, (.9, .1)))
y.unique()
sum(y) / len(y)     # 0.095     !

# grouping var
g = pd.Series(np.random.choice(range(222), N, True))

# some data frame (whatever)
a = np.random.randint(-9, 9, N)
b = np.random.choice(list('abc'), N, True, (.5, .3, .2))
c = np.random.sample(N)
X = pd.DataFrame(dict(a=a, b=b, c=c))
X.head(11)

yg = y.groupby(g)
yg.mean()
sum(yg.mean() > 0)      # 74
sum(yg.mean() > 0) / len(g.unique())    # 0.3348 >> .095

yg.agg(len).value_counts()


plot_covariates(yg.mean(), yg.agg(len), as_factor_x=False, as_factor_y=False)

# train/test split -- stratified and by groups
sgss = StratifiedGroupShuffleSplit(n_splits=1, test_size=.3)
idx_train, idx_test = next(sgss.split(X, y, g))
# !!! remember: it returns numeric indices (as all scikit-learn split generators)
# -- to be used with .iloc to obtain respective data portions

g_train = g.iloc[idx_train]
y_train = y.iloc[idx_train]
g_test = g.iloc[idx_test]
y_test = y.iloc[idx_test]

ug_train = g_train.unique()
ug_test = g_test.unique()

# no common groups
set(ug_train).intersection(ug_test)     # set()  ok

# nr of groups
yg_train = y_train.groupby(g_train)
yg_test = y_test.groupby(g_test)
yg_train.ngroups    # 153
yg_test.ngroups     # 68

yg_test.ngroups / len(g.unique())   # 0.3077  ~= test_size  OK

# nr of  1-groups vs all groups  within train/test subsets
sum(yg_train.sum() > 0)     # 51
sum(yg_test.sum() > 0)      # 23

# in both train/test subsets proportion of 1-groups to all should be roughly tha same
sum(yg_train.sum() > 0) / yg_train.ngroups      # 0.333
sum(yg_test.sum() > 0) / yg_test.ngroups        # 0.338
# which are in turn roughly tha same as
sum(yg.mean() > 0) / len(g.unique())    # 0.3348 >> .095


# %%
def stratified_group_split_summary(y, g, y0, g0):
    X_train = pd.concat([y, g, ['train'] * len(g)], axis=1)
    X_test = pd.concat([y0, g0, ['test'] * len(g0)], axis=1)
    X = pd.concat([X_train,  X_test], axis=0)
    ...
    return X


# %%