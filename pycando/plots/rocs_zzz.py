#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Receiver Operating Characteristic (ROC)
version: 1.0
type: sub-module
keywords: [ROC, AUROC, Gini coefficient]
description: |
content:
remarks:
todo:
sources:
    - title: Receiver Operating Characteristic (ROC)
      link: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc
file:
    name: .py
    date: 2023-09-05
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
import numpy as np
import pandas as pd

import utils.builtin as bi
import utils.plots as pl
from utils.config import pandas_options
pandas_options()

# %%
yy = np.random.randint(0, 2, size=22)
yy
scores = np.random.sample(22)
scores

# pl.helpers.roc()
pl.roc(yy, scores)

# pl.
pl.plot_covariates(yy, scores, what=['boxplots','rocs'])

# %%



# %%