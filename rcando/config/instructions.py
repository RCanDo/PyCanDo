#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Instructions for variable transformations
project: Empirica
version: 1.0
type: config
keywords: [transformations, instructions, directives]
description: |
    Instructions for variable transformations
    to use via  common.transformations.transformers.transform()
remarks:
todo:
sources:
file:
    usage:
        interactive: False   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2022-01-12
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
# import os
import sys
# import json
sys.path.insert(1, "../")

# from functools import partial, update_wrapper

# from common.transformations import tlog1, tlog2, texp1, texp2, power_transformer

# %%
# example lists of transformations
# to be used with
#   common.transformations.transformers.transform()

# TRANSFORMATIONS0 = {
#     'var1':
#         {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
#     'var2':
#         {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
#                # to_factor
#     'var3':
#         {"forward": tlog1, "inverse": texp1, "lower": -70, "upper": 100, "lower_t": None,  "upper_t": None},  # .95
#     }

# TRANSFORMATIONS = {
#     'var1':
#         {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
#     'var2':
#         {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
#               # to_factor
#     'var3':
#         {"forward": tlog1, "inverse": texp1, "lower": None, "upper": 2e4, "lower_t": None,  "upper_t": 10},
#     }

# %%
