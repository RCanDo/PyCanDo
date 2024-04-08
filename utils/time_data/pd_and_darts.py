#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: darts.TimeSeries and pd.DatFrame time data helpers
version: 1.0
type: module
keywords: [time series, attributes, datetime, ...]
description: |
    Helper functions for darts package (darts.TimeSeries data object).
source:
    - title: Time/date components [Pandas]
      link: https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components
    - link: https://unit8co.github.io/darts/generated_api/darts.timeseries.html#darts.timeseries.TimeSeries.add_datetime_attribute  # noqa
todo:
    - smoother by given linear filter wrt to given time interval (attribute like day 'D' or hour 'h'),
      e.g. for data with freq '15T', "aggregate" to freq 'D' then smooth, then expand back to original freq '15T';
      take care of leading and trailing NaNs which arise from this procedure;
file:
    date: 2023-03-01
    authors:
        - nick: edyta
          fullname: Edyta Kania-Strojec
          email:
              - edyta.kania-strojec@quantup.pl
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
import darts as ds
import pandas as pd
from typing import Union

# %%
ATTRIBUTE = {
    "year": lambda ss: ss.dt.year,                  # The year of the datetime
    "month": lambda ss: ss.dt.month,                # The month of the datetime
    "day": lambda ss: ss.dt.day,                    # The days of the datetime
    "hour": lambda ss: ss.dt.hour,                  # The hour of the datetime
    "minute": lambda ss: ss.dt.minute,              # The minutes of the datetime
    "second": lambda ss: ss.dt.second,              # The seconds of the datetime
    "microsecond": lambda ss: ss.dt.microsecond,    # The microseconds of the datetime
    "nanosecond": lambda ss: ss.dt.nanosecond,      # The nanoseconds of the datetime
    "date": lambda ss: ss.dt.date,                  # Returns datetime.date (does not contain timezone information)
    "time": lambda ss: ss.dt.time,                  # Returns datetime.time (does not contain timezone information)
    "timetz": lambda ss: ss.dt.timetz,              # Returns datetime.time as local time with timezone information
    "dayofyear": lambda ss: ss.dt.dayofyear,        # The ordinal day of year
    "day_of_year": lambda ss: ss.dt.day_of_year,    # The ordinal day of year
    "weekofyear": lambda ss: ss.dt.weekofyear,      # The week ordinal of the year
    "week": lambda ss: ss.dt.week,                  # The week ordinal of the year
    "dayofweek": lambda ss: ss.dt.dayofweek,        # The number of the day of the week with Monday=0, Sunday=6
    "day_of_week": lambda ss: ss.dt.day_of_week,    # The number of the day of the week with Monday=0, Sunday=6
    "weekday": lambda ss: ss.dt.weekday,            # The number of the day of the week with Monday=0, Sunday=6
    "quarter": lambda ss: ss.dt.quarter,            # Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.
    "days_in_month": lambda ss: ss.dt.days_in_month,        # The number of days in the month of the datetime
    "is_month_start": lambda ss: ss.dt.is_month_start,      # Logical: is first day of month (defined by frequency)
    "is_month_end": lambda ss: ss.dt.is_month_end,          # Logical: is last day of month (defined by frequency)
    "is_quarter_start": lambda ss: ss.dt.is_quarter_start,  # Logical: is first day of quarter (defined by frequency)
    "is_quarter_end": lambda ss: ss.dt.is_quarter_end,      # Logical: is last day of quarter (defined by frequency)
    "is_year_start": lambda ss: ss.dt.is_year_start,        # Logical: is first day of year (defined by frequency)
    "is_year_end": lambda ss: ss.dt.is_year_end,            # Logical: is last day of year (defined by frequency)
    "is_leap_year": lambda ss: ss.dt.is_leap_year,          # Logical: is the date belongs to a leap year
}


def add_datetime_attributes(
        ts: Union[ds.TimeSeries, pd.DataFrame, pd.Series, pd.DatetimeIndex],
        attributes: tuple[str] = ('year', 'month', 'day', 'hour', 'minute', 'second'),
        one_hot: bool = False,      # only for TimeSeries
        cyclic: bool = False,       # only for TimeSeries
        time_col: str = None,       # only for DataFrame
        time_index: bool = True     # only for Series
) -> Union[ds.TimeSeries, pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Add all datetime `attributes` (passed as list or tuple or single element) to `ts` which may be
    darts.TimeSeries  or pd.DataFrame or pd.Series or pd.DatetimeIndex.
    It's generalised form of `ts.add_datetime_attribute(.)` which can only add one attribute at a time.
    See
    https://unit8co.github.io/darts/generated_api/darts.timeseries.html#darts.timeseries.TimeSeries.add_datetime_attribute
    or
    help(darts.TimeSeries.add_datetime_attribute)
    for more info.
    Type of result depends on the type of input:
        darts.TimeSeries - time data in time_index
            -> darts.TimeSeries - attributes as separate variables (dimensions);
        pd.DataFrame - time data in `time_col` or in index (`time_col` is None)
            -> pd.DataFrame - attributes as separate variables;
        pd.Series - time data in the index (`time_index` True) or in values (`time_index` False)
            -> pd.DataFrame - attributes as separate variables
                (next to values of the pd.Series passed, which may be time data);
                its index is made of time data taken from index or values of the pd.Series passed;
        pd.DatetimeIndex
            -> pd.DataFrame - as above, except there is obviously no column of original values
    """
    if not isinstance(attributes, tuple):
        attributes = (attributes,)

    if isinstance(ts, ds.TimeSeries):
        for a in attributes:
            ts = ts.add_datetime_attribute(a, one_hot, cyclic)
    else:
        if isinstance(ts, pd.DataFrame):
            idx = ts[time_col] if time_col else ts.index.to_series()

        elif isinstance(ts, pd.Series):
            ts = pd.DataFrame(ts)
            if not time_index:
                ts.index = ts.loc[:, 0]
            idx = ts.index.to_series()

        elif isinstance(ts, pd.DatetimeIndex):
            ts = pd.DataFrame([], index=ts)
            idx = ts.index.to_series()

        for a in attributes:
            ts[a] = ATTRIBUTE[a](idx)

    return ts


add_datetime_attribute = add_datetime_attributes    # alias


# %%
def prune_time_index(
        tidx: pd.DatetimeIndex,
        unit: str = 'D',
        closed: tuple[bool] = (True, True)
) -> pd.DatetimeIndex:
    """
    Pruning `tidx` time index to start and end on datetimes rounded wrt to `unit`
    (like day or hour or ...):
        - for start we take ceil of the first date (for given `unit`, e.g. 'D' for day),
        - for end we take floor of the last date.
    tidx: pd.DatetimeIndex,
    unit: str = 'D',
        unit of time wrt prune the series;
    closed: tuple[bool] = (True, True)
        tuple indicating if start and end points (after pruning) belong or not to the final series.
    """
    start = tidx.min().ceil(unit)
    end = tidx.max().floor(unit)

    if closed[0]:
        if closed[1]:
            tidx = tidx[(tidx >= start) & (tidx <= end)]
        else:
            tidx = tidx[(tidx >= start) & (tidx < end)]
    else:
        if closed[1]:
            tidx = tidx[(tidx > start) & (tidx <= end)]
        else:
            tidx = tidx[(tidx > start) & (tidx < end)]

    return tidx

# %%
