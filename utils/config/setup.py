#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Setup for models
version: 1.0
keywords: [setup, libraries, settings, ...]
description: |
    Things common to all model files
file:
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
# import os
# import sys
# sys.path.insert(1, "../")
# sys.path.insert(1, "../../")

import pandas as pd

# from matplotlib.pyplot import figure
# import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
plt.style.use('ggplot')
# plt.style.use('grayscale')
# plt.style.use('dark_background')
# see `plt.style.available` for list of available styles


# %%
def pandas_options(kwargs: dict = {}):
    """
    kwargs: dict = {}
        each `key: value` pair is passed to `pd.set_option(key, value)`.

    https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
    """
    # pd.options.display.width = 0  # autodetects the size of your terminal window - does it work???
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    # pd.options.display.max_rows = 500         # the same
    pd.set_option('display.max_seq_items', None)

    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 3)

    pd.set_option('display.width', 1000)
    pd.set_option('max_colwidth', None)
    # pd.options.display.max_colwidth = 500
    # # the same
    # pd.options.display.width = 120

    for k, v in kwargs.items():
        pd.set_option(k, v)

# pandas_options()

# %%
