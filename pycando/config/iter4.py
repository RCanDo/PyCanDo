#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CONFIGURATION FILE
project: 360e
version: 1.0
type: config
keywords: [parameters, configuration]
description: |
    Mainly
    file names, paths,
    types of variables, groups of variables,
    etc.
    for each data file.
remarks:
    - The convention used here is:
    - parameters used further in code are UPPER_CASE;
    - other elements (like `description`) are lower_case;
todo:
    - data versioning: currently provisional "solution":
      comment other versions, look for  .FILE.RAW=...  in the code
sources:
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2022-08-26
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
# from pathlib import Path
import common.builtin as bi
import common.project as pj

# from attrdict import AttrDict
# from dataclasses import dataclass, field  # fields
# from typing import Any, Dict

# %%
DATA = pj.DataSpec()
# file names
FILE = pj.Files()

FILE.RAW = {
    1: "qu_20221215-20230113.csv",
    11: "inventory_fixed.csv",
    21: "inv_20221212-20221215.csv",
}
FILE.CSV = FILE.RAW

FILE.PREP = {
    1: "qu_20221215-20230113_part_1.pkl",
    2: "qu_20221215-20230113_part_2.pkl",
    3: "qu_20221215-20230113_part_3.pkl",
    4: "qu_20221215-20230113_part_4.pkl",
    11: "inventory_fixed.pkl",
}
FILE.PKL = FILE.PREP

DATA.FILE = FILE

# %% prepared data variables

# metainfo about data
DATA.VAR_OLD_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

DATA.VARS_OLD = bi.union(*DATA.VAR_OLD_TYPES.values())

#  NEW variables AFTER THE LAST DATA UPDATE
#  It serves as a memory of what variables are new wrt. previous data version.
#  When another update happens these should be incorporated into VAR_OLD_TYPES
#  and replaced with another NEW variables.
DATA.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

DATA.VARS_NEW = bi.union(*DATA.VAR_NEW_TYPES.values())

DATA.VAR_TYPES = bi.dict_set_union(DATA.VAR_OLD_TYPES, DATA.VAR_NEW_TYPES)

DATA.VARS_REGISTERED = bi.union(*DATA.VAR_TYPES.values())

# id variables
DATA.VARS_ID = set()

DATA.TARGET = None  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
DATA.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
DATA.VARS_TO_REMOVE = set()

DATA.NOT_PREDICTORS = DATA.VARS_TO_REMOVE.union({DATA.TARGET}).union(DATA.TARGETS).union(DATA.VARS_ID)
DATA.NOT_PREDICTORS = {v for v in DATA.NOT_PREDICTORS if v}     # in case there is NO target

DATA.PREDICTORS = DATA.VARS_REGISTERED.difference(DATA.NOT_PREDICTORS)

# %%  some groups of variable having similar meaning
# DATA.VARS_xxx = set()
# DATA.VARS_yyy = set()
