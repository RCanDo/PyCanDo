#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: holidays
version: 1.0
type: module
keywords: [holidays, datetime index, ...]
description: |
    Holidays wrt to countries
source:
    - title: python-holidays
      link: https://github.com/dr-prodigy/python-holidays#available-countries
todo:
file:
    date: 2023-03-13
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
import pandas as pd
import holidays


def get_holidays(
        tidx: pd.DatetimeIndex,
        country: str = "PL",
        sundays: float = .7,
        saturdays: float = .5,
) -> pd.Series:
    """
    todo:
        - tidx <- pd.Series, pd.DataFrame  x  time in index or values
    """
    hols = holidays.country_holidays(country)
    hols = pd.Series([x in hols for x in tidx], index=tidx).astype(int)
    #
    suns = pd.Series(tidx.to_series().dt.weekday == 6) * sundays
    sats = pd.Series(tidx.to_series().dt.weekday == 5) * saturdays
    #
    result = pd.concat([hols, sats + suns], axis=1).max(axis=1)
    result.index = tidx
    return result
