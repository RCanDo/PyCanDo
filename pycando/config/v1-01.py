#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CONFIGURATION FILE
project: WattStor
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
      comment other versions
sources:
file:
    date: 2022-08-26
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
# from pathlib import Path
# import utils.builtin as bi
import utils.project as pj

# %%
COUNTRIES = {
    'DK': 'CZ',
    'Lučenec': 'SK',
    'Míča-Bagoňová': 'CZ',
    'OC_Dubeň_Žilina': 'SK',
    'Reporeck': 'CZ',
    'SG': 'GB',
    'Uenergy': 'GB',
    'WB_Ciechocinek_backup': 'PL',
    'WattBooster_Ruzomberok': 'SK',
    'ZP_Otice': 'CZ',
    'Miro_Antal': 'SK',
}

# %% DATA ON SITES
# %%

DATA = pj.DataSpec()
# file names

# FILE = pj.Files()     # abandoned (unnecessary complication)
# FILE.RAW = ...
# DATA.FILE = FILE

DATA.FILE_RAW = {
    101: 'SG/SG.csv',
    111: 'Lučenec/Lučenec1.csv',
    112: 'Lučenec/Lučenec2.csv',
    113: 'Lučenec/Lučenec3.csv',
    114: 'Lučenec/Lučenec4.csv',
    115: 'Lučenec/Lučenec5.csv',
    121: 'OC_Dubeň_Žilina/OC_Dubeň_Žilina1.csv',
    122: 'OC_Dubeň_Žilina/OC_Dubeň_Žilina2.csv',
    123: 'OC_Dubeň_Žilina/OC_Dubeň_Žilina3.csv',
    124: 'OC_Dubeň_Žilina/OC_Dubeň_Žilina4.csv',
    131: 'ZP_Otice/ZP_Otice.csv',
    141: 'DK/DK_data.csv',
    # 142: 'DK/DK_data_first_1000_lines.csv',
    151: 'Míča-Bagoňová/Míča-Bagoňová.csv',
    161: 'Uenergy/Uenergy_ws.csv',          # !
    171: 'WB_Ciechocinek_backup/WB_Ciechocinek_backup.csv',
    181: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok1.csv',
    182: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok2.csv',
    183: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok3.csv',
    184: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok4.csv',
    185: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok5.csv',
    186: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok6.csv',
    187: 'WattBooster_Ruzomberok/WattBooster_Ruzomberok7.csv',
    191: 'Reporeck/Reporeck.csv',
    201: 'Miro_Antal/Consumption_part1.csv',
    202: 'Miro_Antal/Consumption_part2.csv',
}
DATA.FILE_CSV = DATA.FILE_RAW

# FILE.PREP = {
DATA.FILE_PREP = {k: file.replace(".csv", ".pkl") for k, file in DATA.FILE_RAW.items()}

DATA.FILE_PKL = DATA.FILE_PREP

DATA.COUNTRIES = COUNTRIES.copy()

# %%
DATA.COL_INDEX = None
DATA.COL_TIME = "time (UTC)"
DATA.COL_SITE = "name"
DATA.COLS_DROP = {
    "time (UTC)",     # drop after it is recreated as 'time' (with proper dtype)
    "tags",           # always empty
    "name", "name(siteID)"    # in one case (161 : Uenergy/Uenergy_data_weather_solar.csv) second version
}
DATA.COL_NAMES = {}   # no name changes needed
DATA.COL_TYPES = {
    "float": ["Consumption", "PV_generation",
              "Grid_consumption", "Grid_backflow",
              "Battery_charging", "Battery_discharging",
              "Holidays"]
}

# %%  some groups of variable having similar meaning
# DATA.VARS_xxx = set()
# DATA.VARS_yyy = set()


# %% WEATHER DATA
# %%

WEATHER = pj.DataSpec()

# file names
WEATHER.FILE_RAW = {
    100: 'SG/SG_weather_solar.csv',
    110: 'Lučenec/Lučenec_weather_solar.csv',
    140: 'DK/DK_weather_solar.csv',
    160: 'Uenergy/Weather_Solar.csv',
    200: 'Miro_Antal/External_temperature_with _weather.csv',
}
WEATHER.FILE_CSV = WEATHER.FILE_RAW

WEATHER.FILE_PREP = {
    100: 'SG/SG_weather_solar.pkl',
    110: 'Lučenec/Lučenec_weather_solar.pkl',
    140: 'DK/DK_weather_solar.pkl',
    160: 'Uenergy/Weather_Solar.pkl',
    200: 'Miro_Antal/External_temperature_with _weather.pkl',
}
WEATHER.FILE_PKL = WEATHER.FILE_PREP

WEATHER.COUNTRIES = COUNTRIES.copy()

# %%

WEATHER.COL_INDEX = 0
WEATHER.COL_TIME = "Time"
WEATHER.COL_SITE = None
WEATHER.COLS_DROP = {
    "Time",     # drop after it is recreated as 'time' (with proper dtype)
}
WEATHER.COL_NAMES = {
    'Temperature, K': 'Temperature_ext',           # Miro_Antal
    'Temper': 'Temperature_ext',
    'pressure': 'Pressure',
    'wind_speed': 'Wind_speed',
    'wind_direction': 'Wind_direction',
    'precipitation': 'Precipitation',
    'cloud_opacity': 'Cloud_opacity',
    'cloud_copacity': 'Cloud_opacity',      # Miro_Antal
    'dni': 'DNI',
    'dhi': 'DHI',
    'ghi': 'GHI',
    'ebh': 'EBH',
}

WEATHER.COL_TYPES = {
    "float": ['Temperature_ext', 'Pressure', 'Wind_speed',
              'Wind_direction', 'Precipitation', 'Cloud_opacity',
              'DNI', 'DHI', 'GHI', 'EBH'],
}

# %% INTERNAL DATA
# %%

INTERNAL = pj.DataSpec()

# file names
INTERNAL.FILE_RAW = {
    201: 'Miro_Antal/Internal_temp_part_1.csv',
    202: 'Miro_Antal/Internal_temp_part_2.csv',
}
INTERNAL.FILE_CSV = INTERNAL.FILE_RAW

INTERNAL.FILE_PREP = {
    201: 'Miro_Antal/Internal_temp_part_1.pkl',
    202: 'Miro_Antal/Internal_temp_part_2.pkl',
}
INTERNAL.FILE_PKL = INTERNAL.FILE_PREP

INTERNAL.COUNTRIES = COUNTRIES.copy()

# %%

INTERNAL.COL_INDEX = None
INTERNAL.COL_TIME = "time (UTC)"
INTERNAL.COL_SITE = "name"
INTERNAL.COLS_DROP = {
    "time (UTC)",     # drop after it is recreated as 'time' (with proper dtype)
    "tags",           # always empty
    "name", "name(siteID)"    # in one case (161 : Uenergy/Uenergy_data_weather_solar.csv) second version
}

INTERNAL.COL_NAMES = {
    # 'Unnamed: 0', 'Time',
    'Internal_Temp, C': 'Temperature_int',
}

INTERNAL.COL_TYPES = {
    "float": ['Temperature_int'],
}

# %%  some groups of variable having similar meaning
# WEATHER.VARS_xxx = set()
# WEATHER.VARS_yyy = set()
