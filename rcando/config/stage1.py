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
EVENTS_PREP_1 = AttrDict()

# some description of data
EVENTS_PREP_1.description = \
    """EVENTS data prepared for modelling;
    it's a result of heavy preprocessing; see:
    analyses/prep_1.py, ...
    """  # E122

# EVENTS_PREP_1.CASES = 65000   #

# # file names
EVENTS_PREP_1.FILE = AttrDict(
    # raw data file -- directly from client
    RAW=None,
    # write prepared data into new file, best .csv (but with ';' separator and without row numbers!)
    PREP=None,
    # notice though that some aspects of data cannot be written to .csv (e.g. data types like dates)
    BIN="events_common_5m_3m.pkl", )

# # prepared data variables

# metainfo about data
EVENTS_PREP_1.VAR_OLD_TYPES = {
    "integer": {'State : Active', 'State : Active End', 'State : Activation', 'State : Activation End',
                'State : Fault', 'State : Reset', 'State : Off', 'State : Turn off', 'State : Enabled',
                'State : Alarm', 'State : Hard alarm', 'State : Fault End', 'State : Alarm End',
                'State : Hard alarm End', 'State : Enabled End', 'State : Reset selective', 'State : Turn on',
                'State : Revision alarm', 'State : Revision alarm End', 'State : Revision End',
                'StateInfo : access level 4', 'StateInfo : automatic', 'StateInfo : automatic, confirmed',
                'StateInfo : unconfirmed', 'StateInfo : External', 'StateInfo : access level 2',
                'StateInfo : access level 8', 'StateInfo : smoke alarm', 'StateInfo : earth fault',
                'CommandSource : Operating panel 1', 'CommandSource : Operating panel 1, User: Userlvl3',
                'CommandSource : Operating panel 1, User: Userlvl2', 'CommandSource : Subcontrol unit 1',
                'CommandSource : Operating panel 1, User: Userlvl8', 'CommandSource : Management system 1',
                'Element : Input', 'Element : Operating panel', 'Element : Output', 'Element : Printer',
                'Element : Detector zone', 'Element : All', 'Element : Loop', 'Element : External',
                'ElementInfo : Internal acoustics', 'ElementInfo : Printer operating panel',
                'ElementInfo : manual callpoint', 'ElementInfo : automatic detector', 'ElementInfo : main-sounder',
                'Target'},
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

EVENTS_PREP_1.VARS_OLD = union(*EVENTS_PREP_1.VAR_OLD_TYPES.values())

#  NEW variables AFTER THE LAST DATA UPDATE
#  It serves as a memory of what variables are new wrt. previous data version.
#  When another update happens these should be incorporated into VAR_OLD_TYPES
#  and replaced with another NEW variables.
EVENTS_PREP_1.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

EVENTS_PREP_1.VARS_NEW = union(*EVENTS_PREP_1.VAR_NEW_TYPES.values())

EVENTS_PREP_1.VAR_TYPES = dict_set_union(EVENTS_PREP_1.VAR_OLD_TYPES, EVENTS_PREP_1.VAR_NEW_TYPES)

EVENTS_PREP_1.VARS_REGISTERED = union(*EVENTS_PREP_1.VAR_TYPES.values())

# id variables
EVENTS_PREP_1.VARS_ID = set()

EVENTS_PREP_1.TARGET = 'Target'  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
EVENTS_PREP_1.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
EVENTS_PREP_1.VARS_TO_REMOVE = {
    'Element : Printer', 'Element : Loop',
    'Element : Connection', 'Element : Management system', 'Element : RemoteAccess',   # not present
    'CommandSource : Operating panel 1', 'CommandSource : Operating panel 1, User: Userlvl3',
    'CommandSource : Operating panel 1, User: Userlvl2', 'CommandSource : Subcontrol unit 1',
    'CommandSource : Operating panel 1, User: Userlvl8', 'CommandSource : Management system 1', }

EVENTS_PREP_1.NOT_PREDICTORS = EVENTS_PREP_1.VARS_TO_REMOVE \
    .union({EVENTS_PREP_1.TARGET}).union(EVENTS_PREP_1.TARGETS).union(EVENTS_PREP_1.VARS_ID)
EVENTS_PREP_1.NOT_PREDICTORS = {v for v in EVENTS_PREP_1.NOT_PREDICTORS if v}     # in case there is NO target

EVENTS_PREP_1.PREDICTORS = EVENTS_PREP_1.VARS_REGISTERED.difference(EVENTS_PREP_1.NOT_PREDICTORS)

# # some groups of variable having similar meaning
# EVENTS_PREP_1.VARS_xxx = set()
# EVENTS_PREP_1.VARS_yyy = set()


# %%
EVENTS_PREP_221003 = AttrDict()

# some description of data
EVENTS_PREP_221003.description = \
    """EVENTS data prepared for modelling;
    it's a result of heavy preprocessing; see:
    analyses/prep_1.py, ...
    """  # E122

# EVENTS_PREP_1.CASES = 65000   #

# # file names
EVENTS_PREP_221003.FILE = AttrDict(
    # raw data file -- directly from client
    RAW=None,
    # write prepared data into new file, best .csv (but with ';' separator and without row numbers!)
    PREP=None,
    # notice though that some aspects of data cannot be written to .csv (e.g. data types like dates)
    BIN={"5m_3m_01": "events_common_5m_3m.pkl",
         "5m_3m_02": "events_common_5m_3m.pkl",
         "5m_30s_01": "events_common_5m_30s.pkl",
         "5m_30s_02": "events_common_5m_30s.pkl",
         "1h_30s_01": "events_common_1h_30s.pkl",
         "1h_5m_01": "events_common_1h_5m.pkl",
         "1d_5m_01": "events_common_1d_5m.pkl", }, )

# # prepared data variables

# metainfo about data
EVENTS_PREP_221003.VAR_OLD_TYPES = {
    "integer": {'State : Activation', 'State : Fault', 'State : Activation End',
                'State : Reset', 'State : Fault End', 'State : Enabled', 'State : Off',
                'State : Turn off', 'State : Turn on', 'State : Off End', 'State : Active',
                'State : Active End', 'State : Enabled End', 'State : Revision alarm',
                'State : Revision alarm End', 'State : Hard alarm', 'State : Alarm',
                'State : Hard alarm End', 'State : Alarm End', 'State : Reset selective',
                'State : Caution', 'State : Caution End', 'State : Intervention',
                'State : Intervention End', 'State : expired', 'State : expired End', 'State : Simulate alarm',
                'StateInfo : Real wire break', 'StateInfo : Start-up', 'StateInfo : automatic',
                'StateInfo : not available', 'StateInfo : access level 2', 'StateInfo : access level 8',
                'StateInfo : temperature alarm', 'StateInfo : smoke alarm', 'StateInfo : External',
                'StateInfo : simulated alarm',
                'CommandSource : Operating panel 1', 'CommandSource : Operating panel 1, User: UĹźytkownik_poz2',
                'CommandSource : Operating panel 1, User: RAJ', 'CommandSource : TZ 1, Slot 1: Output freezed',
                'CommandSource : Detector zone 207', 'CommandSource : Detector zone 101',
                'CommandSource : Detector zone 100',
                'Element : Operating panel', 'Element : Loop', 'Element : Output', 'Element : Detector zone',
                'Element : Printer', 'Element : Module active', 'Element : Input', 'Element : RemoteAccess',
                'Element : All', 'Element : External', 'Element : Intervention', 'ElementInfo : Internal acoustics',
                'ElementInfo : automatic detector', 'ElementInfo : manual callpoint',
                'ElementInfo : Printer operating panel', 'ElementInfo : main-sounder', 'Target'},
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

EVENTS_PREP_221003.VARS_OLD = union(*EVENTS_PREP_221003.VAR_OLD_TYPES.values())

#  NEW variables AFTER THE LAST DATA UPDATE
#  It serves as a memory of what variables are new wrt. previous data version.
#  When another update happens these should be incorporated into VAR_OLD_TYPES
#  and replaced with another NEW variables.
EVENTS_PREP_221003.VAR_NEW_TYPES = {
    "integer": set(),
    "float": set(),
    "date": set(),
    "string": set(),
    "category": set(), }

EVENTS_PREP_221003.VARS_NEW = union(*EVENTS_PREP_221003.VAR_NEW_TYPES.values())

EVENTS_PREP_221003.VAR_TYPES = dict_set_union(EVENTS_PREP_221003.VAR_OLD_TYPES, EVENTS_PREP_221003.VAR_NEW_TYPES)

EVENTS_PREP_221003.VARS_REGISTERED = union(*EVENTS_PREP_221003.VAR_TYPES.values())

# id variables
EVENTS_PREP_221003.VARS_ID = set()

EVENTS_PREP_221003.TARGET = 'Target'  # only ONE variable ! or None if no target

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
EVENTS_PREP_221003.TARGETS = set()

# !!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
EVENTS_PREP_221003.VARS_TO_REMOVE = {
    'Element : Printer', 'Element : Loop',
    'Element : Connection', 'Element : Management system', 'Element : RemoteAccess',   # not present
    'CommandSource : Operating panel 1', 'CommandSource : Operating panel 1, User: Userlvl3',
    'CommandSource : Operating panel 1, User: Userlvl2', 'CommandSource : Subcontrol unit 1',
    'CommandSource : Operating panel 1, User: Userlvl8', 'CommandSource : Management system 1',
    # and
    'CommandSource : Operating panel 1, User: UĹźytkownik_poz2',
    'CommandSource : Operating panel 1, User: RAJ', 'CommandSource : TZ 1, Slot 1: Output freezed',
    'CommandSource : Detector zone 207', 'CommandSource : Detector zone 101',
    'CommandSource : Detector zone 100'}

EVENTS_PREP_221003.NOT_PREDICTORS = EVENTS_PREP_221003.VARS_TO_REMOVE \
    .union({EVENTS_PREP_221003.TARGET}).union(EVENTS_PREP_221003.TARGETS).union(EVENTS_PREP_221003.VARS_ID)
EVENTS_PREP_221003.NOT_PREDICTORS = {v for v in EVENTS_PREP_221003.NOT_PREDICTORS if v}     # in case there is NO target

EVENTS_PREP_221003.PREDICTORS = EVENTS_PREP_221003.VARS_REGISTERED.difference(EVENTS_PREP_221003.NOT_PREDICTORS)

# # some groups of variable having similar meaning
# EVENTS_PREP_221003.VARS_xxx = set()
# EVENTS_PREP_221003.VARS_yyy = set()


# %%
EVENTS = {
    "PREP_1": EVENTS_PREP_1,
    "220813": EVENTS_PREP_1,
    "221003": EVENTS_PREP_221003,
}
