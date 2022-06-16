#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Instructions for transformations
project: Empirica
version: 1.0
type: config             # module, analysis, model, tutorial, help, example, ...
keywords: [transformations, instructions, directives]
description: |
    Instructions for variable transformations.
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

#%%
import os, sys, json
sys.path.insert(1, "../")

from functools import partial, update_wrapper

from utils.transformations import srlog1, srexp1, power_transformer

tlog1 = partial(srlog1, r=1)
update_wrapper(tlog1, srlog1)
tlog1.__name__ = "tlog1"

tlog2 = partial(srlog1, r=2)
update_wrapper(tlog2, srlog1)
tlog2.__name__ = "tlog2"

texp1 = partial(srexp1, r=1)
update_wrapper(texp1, srexp1)
texp1.__name__ = "texp1"

texp2 = partial(srexp1, r=2)
update_wrapper(texp2, srexp1)
texp2.__name__ = "texp2"

#%%
TRANSFORMATIONS0 = {
    'returnDayStd':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'returnStd':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    #'return': {"forward": tlog2, "inverse": texp2},
    'pnlDayStd': {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'pnlStd': {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    #'pnl': {"forward": tlog1, "inverse": None},
    ##
    'alwaysTickBetterOmitValue':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'alwaysTickBetterOmitValueStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.44
    'askPriceLevelsCountMidpoint1MeanStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.35
    'askPriceLevelsCountMidpoint2MeanStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.35
    'askPriceLevelsCountMidpoint3MeanStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.35
    'askQtyPerLevelMean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.38
    'askQtyPerLevelMidpoint1Mean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'askQtyPerLevelMidpoint2Mean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.12
    'askQtyPerLevelMidpoint3Mean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.21
    'askQtySumMidpoint1MeanStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.16
    'askQtySumMidpoint2MeanStd':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'askQtySumMidpoint3MeanStd':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'marketImpactMean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = .14
    'maxMidpointDistanceStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lamda = -.5
    'maxOpenPositionDeviation':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'maxStrategyLoss':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'maxStrategyLossStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.2
    'minCumulativeHedgeStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.34
    'minMidpointDistanceStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -2.81
    'openPositionLimit':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # .01
    'openPositionLimitStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.19
    'orderSizeMultiplier':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'orderSizeStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.67 / to_factor
    'orderToTradeVolumeMean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # .04
    'pricePctChange1d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'pricePctChange7d':
        {"forward": tlog1, "inverse": texp1, "lower": -70, "upper": 100, "lower_t": None,  "upper_t": None},  # .95
    'pricePctChange30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'priceRangeStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -9.21
    'priceSpreadStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.7
    'sniperMinQtyStd':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.27
    'sniperOffset':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'useAdaptiveShift':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'useHedge':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'useSniper':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'volume1dMean30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volume1dMean7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeMax30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeMax7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeMin7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumePctChange1d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumePctChange30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumePctChange7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": 2e4, "lower_t": None,  "upper_t": 10},
    'volumeRange30d':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeRange30dStd':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeRange7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": 9,  "upper_t": None},
    'volumeRange7dStd':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # v
    'volumeSum':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    }

#%%
TRANSFORMATIONS = {
    'returnDayStd':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'returnStd':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    #'return': {"forward": tlog2, "inverse": texp2},
    'pnlDayStd': {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'pnlStd': {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    #'pnl': {"forward": tlog1, "inverse": None},
    ##
    'alwaysTickBetterOmitValue':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'alwaysTickBetterOmitValueStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.44
    'askPriceLevelsCountMidpoint1MeanStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.35
    'askPriceLevelsCountMidpoint2MeanStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.35
    'askPriceLevelsCountMidpoint3MeanStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.35
    'askQtyPerLevelMean':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.38
    'askQtyPerLevelMidpoint1Mean':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'askQtyPerLevelMidpoint2Mean':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.12
    'askQtyPerLevelMidpoint3Mean':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.21
    'askQtySumMidpoint1MeanStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.16
    'askQtySumMidpoint2MeanStd':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'askQtySumMidpoint3MeanStd':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'marketImpactMean':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = .14
    'maxMidpointDistanceStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lamda = -.5
    'maxOpenPositionDeviation':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'maxStrategyLoss':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'maxStrategyLossStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # lambda = -.2
    'minCumulativeHedgeStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.34
    'minMidpointDistanceStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -2.81
    'openPositionLimit':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # .01
    'openPositionLimitStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.19
    'orderSizeMultiplier':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'orderSizeStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.67 / to_factor
    'orderToTradeVolumeMean':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # .04
    'pricePctChange1d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'pricePctChange7d':
        {"forward": power_transformer, "inverse": None, "lower": -70, "upper": 100, "lower_t": -4,  "upper_t": 4},  # .95
    'pricePctChange30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'priceRangeStd':    # market
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -9.21
    'priceSpreadStd':   # market
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.7
    'sniperMinQtyStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # -.27
    'sniperOffset':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'useAdaptiveShift':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'useHedge':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'useSniper':
        {"forward": None, "inverse": None, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},  # to_factor
    'volume1dMean30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volume1dMean7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeMax30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeMax7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeMin7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumePctChange1d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumePctChange30d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumePctChange7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": 2e4, "lower_t": None,  "upper_t": 10},
    'volumeRange30d':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeRange30dStd':
        {"forward": tlog2, "inverse": texp2, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    'volumeRange7d':
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": 9,  "upper_t": None},
    'volumeRange7dStd':
        {"forward": power_transformer, "inverse": None, "lower": None, "upper": None, "lower_t": -4,  "upper_t": None},  # v
    'volumeSum':        # market
        {"forward": tlog1, "inverse": texp1, "lower": None, "upper": None, "lower_t": None,  "upper_t": None},
    }

#%%
