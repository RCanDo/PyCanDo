#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CONFIGURATION FILE
project: Raj
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
    - problem 1
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
from pathlib import Path
from common.builtin import dict_set_union, union
from attrdict import AttrDict

# %% paths

PATH = AttrDict()

PATH.ROOT = Path(__file__).absolute().parents[2].resolve()
PATH.DATA = PATH.ROOT / "data"
PATH.DATA_RAW = PATH.DATA / "raw"
PATH.DATA_PREP = PATH.DATA / "prep"

# !???
PATH.DATA_CSV = PATH.DATA_RAW
# #

PATH.DATA_PKL = PATH.DATA_PREP
# PATH.DATA_PKL = PATH.DATA_PREP

PATH.DATA_CURRENT = PATH.DATA_PREP / "data.pkl"    # .csv preprocessed to .pkl (usualy only proper types)

# %%
# %%  TEMPLATE
#  to be filled for each raw data file

# change the TEMP name to sth informative wrt data in the file
TEMP = AttrDict()

# some description of data
TEMP.description = """"""
# TEMP.CASES = 65000   #

# %% file names
# needs to be filled immediately i.e. BEFORE  new_data_insight.py

TEMP.FILE = AttrDict(       # !? sadly: nested AttrDict() cannot be used like  TEMP.FILE.ATTR = sth  - doesn't work!
    # raw data file -- directly from client
    RAW="_raw_data_file_.csv",
    # may have many minor (but annoying) issues
    # to be fixed BEFORE preprocessing
    # in order to write these data in a more appropriate format (into a separate .csv file)
    # e.g. inconvenient variable (column) names, data not properly aligned, empty variables, encoding, etc.
    # this stage is called 'preparation' and is done with aid of  new_data_insight.py  file.
    # sometimes  there is no need for FILE0 -- data are already in proper format
    #
    # write prepared data into new file, best .csv (but with ';' separator and without row numbers!)
    PREP="_prepared_data_file_.csv",
    #
    # notice though that some aspects of data cannot be written to .csv (e.g. data types like dates)
    # thus we need some binary format (.pkl) to read from it ready to analysis/preprocessing
    BIN="_prepared_data_file_.pkl", )

# %% prepared data variables
# fill in AFTER  new_data_insight.py

# metainfo about data
TEMP.VAR_OLD_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

TEMP.VARS_OLD = union(*TEMP.VAR_OLD_TYPES.values())

# NEW variables AFTER THE LAST DATA UPDATE
# It serves as a memory of what variables are new wrt. previous data version.
# When another update happens these should be incorporated into VAR_OLD_TYPES
# and replaced with another NEW variables.
TEMP.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

TEMP.VARS_NEW = union(*TEMP.VAR_NEW_TYPES.values())

TEMP.VAR_TYPES = dict_set_union(TEMP.VAR_OLD_TYPES, TEMP.VAR_NEW_TYPES)

TEMP.VARS_REGISTERED = union(*TEMP.VAR_TYPES.values())

# id variables -- CANNOT BE PREDICTORS!
# sometimes they can be (like in linear mixed effects models) and then DO NOT put them here
TEMP.VARS_ID = set()

TEMP.TARGET = None  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
TEMP.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!!
#  But it's good for further data preprocessing (before modelling)
TEMP.VARS_TO_REMOVE = set()

TEMP.NOT_PREDICTORS = TEMP.VARS_TO_REMOVE.union({TEMP.TARGET}).union(TEMP.TARGETS).union(TEMP.VARS_ID)
TEMP.NOT_PREDICTORS = {v for v in TEMP.NOT_PREDICTORS if v}     # in case there is NO target

TEMP.PREDICTORS = TEMP.VARS_REGISTERED.difference(TEMP.NOT_PREDICTORS)

# %%  some groups of variable having similar meaning
# TEMP.VARS_xxx = set()
# TEMP.VARS_yyy = set()
