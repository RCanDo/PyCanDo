#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: CONFIGURATION FILE
project: Zooplus
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

# %% DATA
# %%

DATA = pj.DataSpec()
# file names

# FILE = pj.Files()     # abandoned (unnecessary complication)
# FILE.RAW = ...
# DATA.FILE = FILE

DATA.FILE_RAW = {
    101: 'article_sales_light.csv',
    102: 'article_sales_incoherent_group_light.csv',
    103: 'article_sales.csv',
    104: 'ap_meta_data.csv',
    105: 'article_plannig_horizon.csv',
    106: 'delivery_countries.csv',
    107: 'fulfillment_centers.csv',
    #
    111: 'ap_forecast/ap_sales_week.csv',
    112: 'ap_forecast/ap_forecast_2023-01-02.csv',
    113: 'ap_forecast/ap_fs_sales_week.csv',
    114: 'ap_forecast/fs_forecast_2023-01-02.csv',
    #
    121: 'ap_country_forecast/ap_country_sales_week.csv',
    122: 'ap_country_forecast/ap_country_forecast_2023-01-02.csv',
    123: 'ap_country_forecast/ap_country_fs_sales_week.csv',
    124: 'ap_country_forecast/ap_country_fs_forecast_2023-01-02.csv',
    #
    131: 'incoherent_group/incoherent_group_ap_country_fs_forecast.csv',
    132: 'incoherent_group/incoherent_group_fs_sales_week.csv',
    133: 'incoherent_group/incoherent_group_ap_sales_week.csv',
    134: 'incoherent_group/incoherent_group_fs_forecast.csv',
    135: 'incoherent_group/incoherent_group_ap_country_sales_week.csv',
    136: 'incoherent_group/incoherent_group_ap_country_forecast.csv',
    137: 'incoherent_group/incoherent_group_ap_forecast.csv',
    #
    141: 'stock/stock_incoherent_group.csv',
    142: 'stock/stock_normal_group.csv'
}
DATA.FILE_CSV = DATA.FILE_RAW

# FILE.PREP = {
DATA.FILE_PREP = {k: file.replace(".csv", ".pkl") for k, file in DATA.FILE_RAW.items()}

DATA.FILE_PKL = DATA.FILE_PREP

# %%
SPEC = {k: pj.DataSpec() for k in DATA.FILE_RAW.keys()}
DATA.SPEC = SPEC

# %%
file_id = 101   # 'article_sales_light.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "O_id": "o_id",                         # ! order id
    "pc_id": "pc_id",                       # parcel id
    "ap_id": "ap_id",
    "fs_id": "fs_id",
    "fc_id": "fc_id",
    "delivery_country_id": "country_id",    # !
    "site_id": "site_id",
    "order_creation_date": "date_order",    # !
    "parcel_creation_date": "date_parcel",  # !
    "parcel_fulfilment": "date_fulfil",     # !
    "position_amount": "amount",            # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": [
        "o_id", "pc_id",
        "fs_id", "fc_id", "ap_id",
        "country_id", "site_id", "amount"],
    "float": [],
    "date": ["date_order", "date_parcel", "date_fulfil"],
    "string": []  # "pa_name_c"],
}

# %%
file_id = 102   # 'article_sales_incoherent_group_light.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "O_id": "o_id",                         # !
    "pc_id": "pc_id",
    "fs_id": "fs_id",
    "fc_id": "fc_id",
    "ap_id": "ap_id",
    "delivery_country_id": "country_id",    # !
    "site_id": "site_id",
    "order_creation_date": "date_order",    # !
    "parcel_creation_date": "date_parcel",  # !
    "parcel_fulfilment": "date_fulfil",     # !
    "position_amount": "amount",            # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": [
        "o_id", "pc_id",
        "fs_id", "fc_id", "ap_id",
        "country_id", "site_id", "amount"],
    "float": [],
    "date": ["date_order", "date_parcel", "date_fulfil"],
    "string": []  # "pa_name_c"],
}

# %%
file_id = 103   # 'article_sales.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = [
    "pa_name_c",
    "pa_level1_id",
    "pa_level2_id",
    "pa_level3_id",
    "pa_level4_id",
    "pa_level5_id",
    "pa_mpp_net1_n",
    "pa_weight_g_n",
    "lai_rank",
]
SPEC[file_id].COL_NAMES = {
    # original: new
    "fs_id": "fs_id",
    "fc_id": "fc_id",
    "ap_id": "ap_id",
    # "pa_name_c": "pa_name_c",
    # "pa_level1_id": "pa_level1_id",
    # "pa_level2_id": "pa_level2_id",
    # "pa_level3_id": "pa_level3_id",
    # "pa_level4_id": "pa_level4_id",
    # "pa_level5_id": "pa_level5_id",
    # "pa_mpp_net1_n": "pa_mpp_net1_n",
    # "pa_weight_g_n": "pa_weight_g_n",
    # "lai_rank": "lai_rank",
    "delivery_country_id": "country_id",    # !
    "site_id": "site_id",
    "position_amount": "amount",            # !
    "order_creation_date": "date_order",    # !
    "parcel_creation_date": "date_parcel",  # !
    "parcel_fulfilment": "date_fulfil",     # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": [
        "fs_id", "fc_id", "ap_id",
        # "pa_level1_id", "pa_level2_id", "pa_level3_id", "pa_level4_id", "pa_level5_id",
        # "lai_rank",
        "country_id", "site_id", "amount"],
    "float": [],    # "pa_mpp_net1_n", "pa_weight_g_n",
    "date": ["date_order", "date_parcel", "date_fulfil"],
    "string": []  # "pa_name_c"],
}

# %%
file_id = 104   # 'ap_meta_data.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = [
    "ap_id", "pa_name_c",
    "pa_level1_id", "pa_level2_id", "pa_level3_id", "pa_level4_id",
    "article_rank", "pa_mpp_net1_n", "pa_weight_g_n"
]
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "pa_level1_id", "pa_level2_id", "pa_level3_id", "pa_level4_id", "article_rank"],
    "float": ["pa_mpp_net1_n", "pa_weight_g_n"],
    "date": [],
    "string": ["pa_name_c"]
}

# %%
file_id = 105   # 'article_plannig_horizon.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = ["ap_id", "ph_weeks"]
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "ph_weeks"],
    "float": [],
    "date": [],
    "string": []
}

# %%
file_id = 106   # 'delivery_countries.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "c_id": "country_id",                   # !
    "c_name": "country",                    # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["country_id"],
    "float": [],
    "date": [],
    "string": ["country"]
}

# %%
file_id = 107   # 'fulfillment_centers.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "fc_id": "fc_id",
    "fc_short_name": "fc_short_name",
    "fc_country": "fc_country",
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["fc_id"],
    "float": [],
    "date": [],
    "string": ["fc_short_name", "fc_country"]
}

# %% ------------------------------------------------------------------------------------------------------------------
file_id = 111   # 'ap_forecast/ap_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "AP_ID": "ap_id",                       # !
    "SALES_WEEK": "week",                   # !
    "ORIGINAL_SALES": "sales",              # !
    "COMPENSATED_SALES": "sales_c",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id"],
    "float": ["sales", "sales_c"],
    "date": ["week"]
}

# %%
file_id = 112   # 'ap_forecast/ap_forecast_2023-01-02.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "ap_id": "ap_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 113   # 'ap_forecast/ap_fs_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "sales_week": "week",                   # !
    "sales_quantity": "sales",              # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["sales"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 114   # 'ap_forecast/fs_forecast_2023-01-02.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "fs_id": "fs_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id", "fs_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %% ------------------------------------------------------------------------------------------------------------------
file_id = 121   # 'ap_country_forecast/ap_country_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "ap_id": "ap_id",
    "country_id": "country_id",
    "sales_week": "week",
    "original_sales": "sales",              # !
    "compensated_sales": "sales_c",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "country_id"],
    "float": ["sales", "sales_c"],
    "date": ["week"]
}

# %%
file_id = 122   # 'ap_country_forecast/ap_country_forecast_2023-01-02.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "country_id": "country_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "country_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 123   # 'ap_country_forecast/ap_country_fs_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "sales_week": "week",                   # !
    "sales_quantity": "sales",              # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["sales"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 124   # 'ap_country_forecast/ap_country_fs_forecast_2023-01-02.csv',
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %% ------------------------------------------------------------------------------------------------------------------
file_id = 131   # 'incoherent_group/incoherent_group_ap_country_fs_forecast.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 132   # 'incoherent_group/incoherent_group_fs_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "sales_week": "week",                   # !
    "sales_quantity": "sales",              # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["sales"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 133  # 'incoherent_group/incoherent_group_ap_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    # original: new
    "AP_ID": "ap_id",                       # !
    "SALES_WEEK": "week",                   # !
    "ORIGINAL_SALES": "sales",              # !
    "COMPENSATED_SALES": "sales_c",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id"],
    "float": ["sales", "sales_c"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 134  # 'incoherent_group/incoherent_group_fs_forecast.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "fs_id": "fs_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id", "fs_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 135  # 'incoherent_group/incoherent_group_ap_country_sales_week.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "country_id": "country_id",
    "sales_week": "week",                   # !
    "original_sales": "sales",              # !
    "compensated_sales": "sales_c",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "country_id"],
    "float": ["sales", "sales_c"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 136  # 'incoherent_group/incoherent_group_ap_country_forecast.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "country_id": "country_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "country_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 137   # 'incoherent_group/incoherent_group_ap_forecast.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "ap_id": "ap_id",
    "forecast_week": "week",                # !
    "forecast_quantity": "sales_f",         # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id"],
    "float": ["sales_f"],
    "date": ["week"],
    "string": []
}

# %%
file_id = 141   # 'stock/stock_incoherent_group.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "inventory_date": "date",  # !
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "stock_quantity": "stock",  # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["stock"],
    "date": ["date"],
    "string": []
}

# %%
file_id = 142   # 'stock/stock_normal_group.csv'
SPEC[file_id] = pj.DataSpec()
# before renaming -- original names
SPEC[file_id].COLS_DROP = []
SPEC[file_id].COL_NAMES = {
    "inventory_date": "date",  # !
    "ap_id": "ap_id",
    "fc_id": "fc_id",
    "stock_quantity": "stock",  # !
}
# after renaming
SPEC[file_id].COL_TYPES = {
    "integer": ["ap_id", "fc_id"],
    "float": ["stock"],
    "date": ["date"],
    "string": []
}

# %%  some groups of variable having similar meaning
# DATA.VARS_xxx = set()
# DATA.VARS_yyy = set()
