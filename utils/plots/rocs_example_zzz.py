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

from utils.config import pandas_options
pandas_options()
from utils.project import Paths
import utils.df as udf
import utils.plots as pl


# %% load ready example data
df3 = pickle.load(open("../../../Data/auroc_data.pkl", 'rb'))
udf.info(df3)

variable = df3['bid_price']
factor = df3["is_bid_won"].astype('category')

df3.groupby("is_bid_won").agg(np.mean)
df3["is_bid_won"].value_counts()
df3.groupby("is_bid_won").agg([np.mean, np.std])

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