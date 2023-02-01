#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 18:25:33 2023

@author: arek
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import dill as pickle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from matplotlib.colors import ListedColormap  # LinearSegmentedColormap

from common.config import pandas_options
pandas_options()
from common.project import Paths
import common.adhoc as ah
import common.df as cdf
import common.plots as pl

# %% getting example data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')

    df['is_clicked'] = df["pod3.clicks"] > 0
    df['is_purchased'] = df["purchases.purch_ts"] > 0
    df['is_bid_won'] = df["cid_imp"] > 0

    df['bids_datetime'] = cdf.to_datetime(df["bids.ts"], unit='s', floor='s')
    df['purchase_datetime'] = cdf.to_datetime(df["purchases.purch_ts"], unit='s', floor='s')
    df['min_click_datetime'] = cdf.to_datetime(df["pod3.min_click_ts"], unit='s', floor='s')

    return df

PATH = Paths('iter4/kampania_ustalania_ceny')

df = ah.load(file=1, path=PATH.DATA_PREP, what=['stem'])
df = df['stem']

df = prepare_data(df)
df.shape    # (249999, 28)
cdf.info(df)

df2 = df[['bids.bid_price', "is_bid_won", "bids.boost", "bids.cps"]].dropna()
df2.rename(columns={"bids.bid_price": "bid_price", "bids.boost": "boost", "bids.cps": "cps"}, inplace=True)
df3 = df2.sample(int(2e4))
cdf.info(df3)
df3.to_pickle("zzz/auroc_data.pkl")

# %% load ready example data
df3 = pickle.load(open("zzz/auroc_data.pkl", 'rb'))

variable = df3['bid_price']
factor = df3["is_bid_won"].astype('category')

df3.groupby("is_bid_won").agg(np.mean)

# %%
fpr, tpr, thresh = roc_curve(factor.astype(int), variable)

fpr, tpr, thresh, auroc = pl.rocs(variable, factor)

cats, cat_colors, cmap = pl.cats_and_colors(factor)
cats
cat_colors

fig, ax = plt.subplots()
fpr, tpr, thresh, auroc = pl.plot_rocs(ax, variable, factor, cats, cat_colors, cmap)

pl.plot_covariates(variable, factor.astype(int), what=['grouped_cloud', 'densities','rocs'])
roc_auc_score(factor.astype(int), variable)
auc(fpr, tpr)

# %%
variable = df3['bid_price']
variable2 = df3['cps']
factor1 = df3["is_bid_won"]
factor2 = df3["boost"]

res = pl.plot_covariates(variable, factor1, what=['grouped_cloud', 'densities','rocs'], res=True, n_obs=None)
res = pl.plot_covariates(variable, factor1, what=['grouped_cloud', 'densities','rocs'], res=True, n_obs=1000)
res = pl.plot_covariates(variable, factor1, what=['grouped_cloud', 'densities','rocs'], res=True)
res.keys()
res['title']
res['plot'].keys()
res['plot']['agg']
res['plot']['rocs'].keys()
res['plot']['rocs']['result'].keys()
res['plot']['rocs']['result']['thresh']

res = pl.plot_covariates(variable, factor1, what=[['grouped_cloud', 'rocs'], ['densities', 'boxplots'], ['distr', 'blank']], res=True)

# %%
res = pl.plot_covariates(variable, factor2, res=True)
ax = res['plot']['densities']['ax']
dir(ax)
dir(ax.legend)
help(ax.legend)

# %%
res = pl.plot_covariates(variable, factor2, what=['grouped_cloud', 'densities','rocs'], res=True)

# %%
res = pl.plot_covariates(variable, variable2, res=True)
res = pl.plot_covariates(variable, variable2, res=True, n_obs=1000)
res = pl.plot_covariates(variable, variable2, res=True, alpha=1)
res.keys()
res['plot'].keys()
res['plot']['cloud'].keys()
res['plot']['cloud']['axes'].keys()
res['plot']['cloud']['result'].keys()
res['plot']['cloud']['result']['scatter_hist'].keys()
res['plot']['cloud']['result']['smoother'].keys()

res = pl.plot_covariates(variable, variable2, res=True, alpha=1, smooth=False)
res['plot']['cloud']['result'].keys()
res['plot']['cloud']['result']['smoother'].keys()
res = pl.plot_covariates(variable, variable2, res=True, alpha=1, smooth=8)

# %%
res = pl.plot_variable(variable, res=True)
res = pl.plot_variable(variable, res=True, n_obs=1000)
res.keys()
res['plot'].keys()
res['plot']['axs']
res['plot']['axs'][0,0]

pl.plot_variable(factor1)
pl.plot_variable(factor2)


# %%