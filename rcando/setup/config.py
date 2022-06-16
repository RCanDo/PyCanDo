#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: EMPIRICA CONFIGURATION FILE
project: Empirica
version: 1.0
type: config             # module, analysis, model, tutorial, help, example, ...
keywords: [parameters, configuration]
description: |
    Empirica configuration file.
    Mainly types of variables, groups of variables, paths.
remarks:
    - etc.
todo:
    - This file should be in a different folder, but...
sources:
    - title:
      chapter:
      pages:
      link: https://the_page/../xxx.domain
      date:    # date of issue or last edition of the page
      authors:
          - nick:
            fullname:
            email:
      usage: |
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2022-02-01
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
#%%
from pathlib import Path
from utils.builtin import dict_set_union, union

ROOT = Path(__file__).absolute().parents[2].resolve()
PATH_DATA = ROOT / "data/"
PATH_DATA_CSV = PATH_DATA / "source_data/current_data.csv"      # this is soft link to the latest 'official' data version .csv
PATH_OUTPUT = PATH_DATA / 'prep_data/'
PATH_DATA_PKL = PATH_DATA / "prep_data/data_prep.pkl"           # PATH_DATA_CSV preprocessed to .pkl (usualy only proper types )

#PATH_DATA = Path(__file__).absolute().parent.parent / 'data/source_data/newest_data'
#os.makedirs(PATH_OUTPUT, exist_ok=True)
CASES = 6576

# metainfo about data
COL_TYPES = {
    "integer": {"durationSeconds", "maxOpenPositionDeviation", "delayReject", "delayTrade", "noTicks",
            "hedgeOrderRefreshTimeMilliSeconds", "hedgeTickWorseNo", "changeEnd", "changeStart"},
    "float": {"fee", "volumeSum", "priceMean", "priceLow", "priceHigh", "priceSpreadStd", "priceSpread",
            "priceRange", "priceRangeStd", "openPositionLimit", "orderSize", "orderSizeMultiplier",
            "maxStrategyLoss", "marketDepth", "marketDepthStd", "marketDepthBenchmark", "minMidpointDistance",
            "maxMidpointDistance", "alwaysTickBetterOmitValue", "sniperOffset", "sniperMinQty",
            "minCumulativeHedge", "adaptiveShiftMinMinMidpointDistance", "openPositionLimitStd",
            "adaptiveShiftPercentileHigher", "adaptiveShiftPercentileLower",
            "adaptiveShiftTimeFrameDuration", "adaptiveShiftTimeFramesCount", "pnl", "return",
            "sniperMinQtyStd", "hedgeFee", "alwaysTickBetterOmitValueStd", "orderSizeStd",
            "maxStrategyLossStd", "askPriceMean", "minCumulativeHedgeStd", "bidPriceMean"},
    "date": {"time"},
    "string": {"instanceId"},
    "category": {"instrument", "exchange", "strategyMode", "isNoTakerOrders", "ignoreLastTrade",
            "useSniper", "useHedge", "useAdaptiveShift"}
    }

VARS_OLD = union(*COL_TYPES.values())

## NEW variables AFTER THE LAST DATA UPDATE
## It serves as a memory of what variables are new wrt. previous data version.
## When another update happens these should be incorporated into COL_TYPES (as OLD vairables now)
## and replaced with another NEW variables.
COL_NEW_TYPES = {
    "float": {'orderQtySum', 'tradeQtySum', 'orderCount', 'tradeCount',
        'askQtyPerLevelMean', 'askQtyPerLevelMidpoint1Mean', 'volumeMax7d', 'volumeMin7d', 'volumeRange7dStd',
        'returnStd', 'askPriceLevelsCountMidpoint1Mean', 'askQtySumMidpoint1MeanStd', 'askQtySumMidpoint2Mean',
        'volumeRange30dStd', 'askQtySumMidpoint3Mean', 'orderToTradeVolumeMean', 'volumeRange7d', 'returnVolumeStd',
        'askPriceLevelsCountMidpoint1MeanStd', 'askPriceLevelsCountMidpoint2MeanStd', 'askQtySumMean',
        'volumeMax30d', 'volumePctChange30d', 'volume1dMean30d', 'askQtyPerLevelMidpoint3Mean', 'pnlVolumeStd',
        'volumeRange30d', 'askPriceLevelsCountMidpoint2Mean', 'askQtyPerLevelMidpoint2Mean', 'biasMean',
        'pricePctChange7d', 'askPriceLevelsCountMidpoint3Mean', 'pnlStd', 'askQtySumMidpoint2MeanStd',
        'seriesDurationSeconds', 'volumePctChange7d', 'askPriceLevelsCountMidpoint3MeanStd', 'pricePctChange1d',
        'askQtySumMidpoint3MeanStd', 'askQtySumMidpoint1Mean', 'volume1dMean7d', 'volumePctChange1d', 'pricePctChange30d',
        'marketImpactMean',
        # newest
        'returnDayStd', 'pnlDayStd',
        'minMidpointDistanceStd', 'maxMidpointDistanceStd',
        'seriesDaysDurationSeconds',
        # "maxMidpoiontDistance" # not in data! -- just a typo!!! :)
        #'change_id'
        }
    }

VARS_NEW = union(*COL_NEW_TYPES.values())

COL_TYPES = dict_set_union(COL_TYPES, COL_NEW_TYPES)

VARS_REGISTERED = union(*COL_TYPES.values())

VARS_ID = {'instanceId', 'exchange', 'instrument'}

TARGET = 'returnDayStd'

# POSSIBLE ALTERNATIVE TARGETS -- CANNOT BE PREDICTORS!
TARGETS = {'pnl', 'pnlStd', 'pnlDayStd', 'return', 'returnStd'}

#!!! DON'T USE IT DIRECTLY!!! i.e. don't use `clear` option in functions below.
#  But it's good for further data preprocessing (before modelling)
VARS_TO_REMOVE = {
    "biasMean", "askQtySumMean",
    "pnl", "return",
    # cannot be predictors
    'pnlVolumeStd', 'returnVolumeStd',
    'tradeQtySum', 'tradeCount',
    'time', 'changeStart', 'changeEnd',
    #'instanceId',   # there always should be some ID variable(s)
                     # e.g. it's imporant for excluding some special instances
    # empty or 1 value:
    'noTicks', 'isNoTakerOrders', 'ignoreLastTrade', 'hedgeTickWorseNo',
    'adaptiveShiftPercentileHigher', 'adaptiveShiftPercentileLower',
    # we just don't want it:
    'askPriceMean',
    'askPriceLevelsCountMidpoint1Mean', 'askPriceLevelsCountMidpoint2Mean',
    'askPriceLevelsCountMidpoint3Mean',
    'askQtySumMidpoint1Mean', 'askQtySumMidpoint2Mean', 'askQtySumMidpoint3Mean',
    'adaptiveShiftMinMinMidpointDistance',
    'adaptiveShiftTimeFrameDuration',  'adaptiveShiftTimeFramesCount',
    'bidPriceMean',
    'delayReject',  'delayTrade',  'durationSeconds',
    'hedgeOrderRefreshTimeMilliSeconds',
    'marketDepthStd',  'marketDepth',  'marketDepthBenchmark',
    'minCumulativeHedge',
    "minMidpointDistance",
    "maxMidpointDistance",
    'orderCount',  'orderQtySum',
    'priceSpread', 'priceMean', 'priceLow', 'priceHigh', 'priceRange',
    'seriesDurationSeconds',  'seriesDaysDurationSeconds',
    'strategyMode',  'sniperMinQty',
    "orderSize",
    # "askPriceLevelsCountMean", 'volumeMin30d',  #!? not in data
    "fee", "hedgeFee", "delayReject",
    #
    'volume1dMean7d', 'volumeMin7d',  'volumeMax7d', 'volumeMax30d',
    'volumeRange7d', 'volumeRange30d', 'volumeRange30dStd',
    'pricePctChange1d', 'pricePctChange30d', 'volumePctChange1d', 'volumePctChange30d',
    "maxStrategyLoss", "maxStrategyLossStd",
    "minCumulativeHedge"
    }

NOT_PREDICTORS = VARS_TO_REMOVE.union({TARGET}).union(VARS_ID).union(TARGETS)

PREDICTORS = VARS_REGISTERED.difference( NOT_PREDICTORS )

#%%
VARS_MARKET = {'volumeSum', 'priceSpreadStd', 'priceRangeStd',
    "volumePctChange7d", "pricePctChange7d", "volumeRange7dStd", "volume1dMean30d"}

VARS_ALGORITHM = {'openPositionLimit', 'orderSize', 'useHedge', "useSniper",
    "minMidpointDistanceStd", "maxMidpointDistanceStd", "sniperOffset", "useAdaptiveShift",
    "maxOpenPositionDeviation"}

#%%
