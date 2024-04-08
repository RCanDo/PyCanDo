#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CONFIGURATION FILE
project:
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
from common.builtin import dict_set_union, union
from attrdict import AttrDict

# %%
# %%
EVENTS = AttrDict()

# some description of data
EVENTS.description = \
    """multivariate time-series with events;
    no time index (evenly spaced) but 'Date' variable (Timestamp)
    indicating time the event happend;
    many events in one time-point possible;
    """  # E122

# EVENTS.CASES = 65000   #

# %% file names
EVENTS.FILE = AttrDict(
    # raw data file -- directly from client
    RAW={"220813": "Event memory_SCU1_20220813_072459_1.csv",
         "221003": "20221003pz_Bielsko_1.csv"},
    # write prepared data into new file, best .csv (but with ';' separator and without row numbers!)
    PREP="events.csv",
    # notice though that some aspects of data cannot be written to .csv (e.g. data types like dates)
    BIN="events.pkl",
    ALARMS="alarms_derivatives.pkl"
)

# %% prepared data variables

# metainfo about data
EVENTS.VAR_OLD_TYPES = {
    "integer": {'CurrentNumber'},
    "float": set(),
    "date": {'Date'},
    "string": {'Element', 'ElementInfo', 'Number', 'State', 'StateInfo', 'CommandSource'},
    "category": set(), }

EVENTS.VARS_OLD = union(*EVENTS.VAR_OLD_TYPES.values())

#  NEW variables AFTER THE LAST DATA UPDATE
#  It serves as a memory of what variables are new wrt. previous data version.
#  When another update happens these should be incorporated into VAR_OLD_TYPES
#  and replaced with another NEW variables.
EVENTS.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

EVENTS.VARS_NEW = union(*EVENTS.VAR_NEW_TYPES.values())

EVENTS.VAR_TYPES = dict_set_union(EVENTS.VAR_OLD_TYPES, EVENTS.VAR_NEW_TYPES)

EVENTS.VARS_REGISTERED = union(*EVENTS.VAR_TYPES.values())

# id variables
EVENTS.VARS_ID = {'CurrentNumber', 'Date'}

EVENTS.TARGET = None  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
EVENTS.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
EVENTS.VARS_TO_REMOVE = set()

EVENTS.NOT_PREDICTORS = EVENTS.VARS_TO_REMOVE.union({EVENTS.TARGET}).union(EVENTS.TARGETS).union(EVENTS.VARS_ID)
EVENTS.NOT_PREDICTORS = {v for v in EVENTS.NOT_PREDICTORS if v}     # in case there is NO target

EVENTS.PREDICTORS = EVENTS.VARS_REGISTERED.difference(EVENTS.NOT_PREDICTORS)

# %%  some groups of variable having similar meaning
# EVENTS.VARS_xxx = set()
# EVENTS.VARS_yyy = set()


# %%
# %%  OUTER

OUTER = AttrDict()

# OUTER.CASES =    #

# some description of data
OUTER.description = \
    """proper time-series (evenly-spaced)
    with outer temperature
    """

# %% file names
# %% file names
OUTER.FILE = AttrDict(
    # raw data file -- directly from client
    RAW="1_1_1_10_AI_2796310.csv",
    # write prepared data into new file, best .csv (but with ';' separator and without row numbers!)
    PREP="temperature_outer.csv",
    # notice though that some aspects of data cannot be written to .csv (e.g. data types like dates)
    BIN="temperature_outer.pkl", )

# %% prepared data variables

# metainfo about data
OUTER.VAR_OLD_TYPES = {
    "integer": set(),
    "float": {"Temperature"},
    "date": {"Date"},
    "string": set(),
    "category": set(), }

OUTER.VARS_OLD = union(*OUTER.VAR_OLD_TYPES.values())

#  NEW variables AFTER THE LAST DATA UPDATE
#  It serves as a memory of what variables are new wrt. previous data version.
#  When another update happens these should be incorporated into VAR_OLD_TYPES
#  and replaced with another NEW variables.
OUTER.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

OUTER.VARS_NEW = union(*OUTER.VAR_NEW_TYPES.values())

OUTER.VAR_TYPES = dict_set_union(OUTER.VAR_OLD_TYPES, OUTER.VAR_NEW_TYPES)

OUTER.VARS_REGISTERED = union(*OUTER.VAR_TYPES.values())

# id variables
OUTER.VARS_ID = {'Date'}

OUTER.TARGET = None  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
OUTER.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
OUTER.VARS_TO_REMOVE = set()

OUTER.NOT_PREDICTORS = OUTER.VARS_TO_REMOVE.union({OUTER.TARGET}).union(OUTER.TARGETS).union(OUTER.VARS_ID)
OUTER.NOT_PREDICTORS = {v for v in OUTER.NOT_PREDICTORS if v}     # in case there is NO target

OUTER.PREDICTORS = OUTER.VARS_REGISTERED.difference(OUTER.NOT_PREDICTORS)

# %%  some groups of variable having similar meaning
# OUTER.VARS_xxx = set()
# OUTER.VARS_yyy = set()


# %%
# %%  INNER

INNER = AttrDict()

# OUTER.CASES =    #

# some description of data
INNER.description = \
    """proper time-series (evenly-spaced)
    with inner temperatures in 4 locations
    """

# %% file names
INNER.FILE = AttrDict(
    # raw data file -- directly from client
    RAW="Protokol roboczy.csv",
    # write prepared data into new file, best .csv (but with ';' separator and without row numbers!)
    PREP="temperature_inner.csv",
    # notice though that some aspects of data cannot be written to .csv (e.g. data types like dates)
    BIN="temperature_inner.pkl", )

# %% prepared data variables

# metainfo about data
INNER.VAR_OLD_TYPES = {
    "integer": set(),
    "float": {'Temperature_2', 'Temperature_3', 'Temperature_4', 'Temperature_1'},
    "date": {'Date'},
    "string": set(),
    "category": set(), }

INNER.VARS_OLD = union(*INNER.VAR_OLD_TYPES.values())

# NEW variables AFTER THE LAST DATA UPDATE
# It serves as a memory of what variables are new wrt. previous data version.
# When another update happens these should be incorporated into VAR_OLD_TYPES
# and replaced with another NEW variables.
INNER.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

INNER.VARS_NEW = union(*INNER.VAR_NEW_TYPES.values())

INNER.VAR_TYPES = dict_set_union(INNER.VAR_OLD_TYPES, INNER.VAR_NEW_TYPES)

INNER.VARS_REGISTERED = union(*INNER.VAR_TYPES.values())

# id variables
INNER.VARS_ID = {'Date'}

INNER.TARGET = None  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
INNER.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
INNER.VARS_TO_REMOVE = set()

INNER.NOT_PREDICTORS = INNER.VARS_TO_REMOVE.union({INNER.TARGET}).union(INNER.TARGETS).union(INNER.VARS_ID)
INNER.NOT_PREDICTORS = {v for v in INNER.NOT_PREDICTORS if v}     # in case there is NO target

INNER.PREDICTORS = INNER.VARS_REGISTERED.difference(INNER.NOT_PREDICTORS)

# %%  some groups of variable having similar meaning
# INNER.VARS_xxx = set()
# INNER.VARS_yyy = set()
