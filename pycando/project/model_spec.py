#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: model specification class
version: 1.0
type: module
keywords: [model, metrics, ...]
description: |
    Similar to R, we need model container which provides all the facilities
    - for storing meta-information (like configs, params, paths, etc.),
    - for validation and model diagnosis;
    - for reporting.
remarks:
    - Currently abandoned for the sake of  utils/stats/model_diag.py  which is better managable
      and more precisely specified:
      statistical diagnostics of the model identified via some Storage
      (all info necessary for loading model and relative data).
todo:
    - EVERYTHING
sources:
file:
    date: 2022-05-22
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
# import json
# from pathlib import Path
# from functools import partialmethod

# import numpy as np
# import pandas as pd

# # from scipy.stats import entropy
# from sklearn.metrics import r2_score, f1_score, accuracy_score  # roc_auc_score, roc_curve,
# from sklearn.metrics import mean_squared_error, confusion_matrix, explained_variance_score
# from sklearn.model_selection import cross_val_score

# # from utils.builtin import flatten, coalesce
import utils.builtin as bi

# from .helpers import to_dict

# from dataclasses import dataclass, field  # fields
# from typing import Any, List, Dict, Set  # , Iterable
# from copy import deepcopy


# %%
class ModelSpec(bi.Repr):

    def __init_(
        self,
        model,
        storage,
    ):
        """"""
        self.model = model
        self.storage = storage
        self.config = self.get_config()

    def get_config(self):
        """"""
        self.config = self.storage.load('config')

    def update_storage(self, storage):
        """"""
