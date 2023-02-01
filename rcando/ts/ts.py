#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: date/time helpers
version: 1.0
type: module             # module, analysis, model, tutorial, help, example, ...
keywords: [time series, datetime, aggregation,]
description: |
remarks:
todo:
file:
    usage:
        interactive: False  # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2022-09-13
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
from typing import Union, Tuple
# import numpy as np
import pandas as pd


# %%
class TimeInterval():
    """
    straightforward alternative for pd.Period() which is awkward in use;
    Has method __contains__ relevant to `in` operator
    for checking if a given moment (datetime) belongs to time interval;
    also it's possible to set up if the start and end points belong to the interval.

    Examples
    --------

    iv = TimeInterval('2022-01-01', '2 D', closed=(1, 0))
    iv              # 2022-01-01T00:00:00 - 2022-01-03T00:00:00 (2 days 00:00:00)
    iv.start in iv  # True
    iv.end in iv    # False

    t = pd.Timestamp('2022-01-02 23:59:59')
    t in iv         # True

    iv = TimeInterval('2022-01-01', '-2 D', closed=(0, 1))
    iv              # 2021-12-30T00:00:00 - 2022-01-01T00:00:00 (2 days 00:00:00)
    t in iv         # False
    iv.start in iv  # False
    iv.end in iv    # True

    t2 = pd.Timestamp('2022-01-01')
    t2 in iv        # True
    """

    def __init__(
            self,
            start: Union[pd.Timestamp, str],
            end: Union[pd.Timestamp, pd.Timedelta, str] = None,
            closed: Tuple[bool, bool] = (True, True), ):
        """
        start : str, pd.Timestamp, pd.Timedelta;
        end : str, pd.Timestamp, pd.Timedelta;
            if end is pd.Timedelta (or relative str) then the proper end of interval
            is derived from it based on start;
        Finally, always  self.start <= self.end  !!!
        and  self.timedelta = self.start - self.end >= 0;
        closed : (bool, bool) : (True, True);
            consider respective ends of the `interval = (start, end)`
            as belonging to it (True) or not (False);
            i.e. 'True' is equiv. to '[' or ']' and 'False' to '(' or ')' in math notation.

        See here: https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
        for possible str formats to pass timedelta,
        e.g. '3 T' means 3 minutes, '3 M' is 3 months;
        """
        if isinstance(start, str):
            start = pd.Timestamp(start)

        if not end:
            end = start

        if isinstance(end, str):
            try:
                end = pd.Timestamp(end)
            except ValueError:
                end = pd.Timedelta(end)

        if isinstance(end, pd.Timedelta):
            end = start + end

        self.start, self.end = sorted([start, end])

        self.timedelta = self.end - self.start

        if self.timedelta.value == 0:
            # one point interval
            self.closed = (True, True)

        self.closed = closed

    def __getattr__(self, attr):
        return getattr(self.timedelta, attr)

    def __contains__(self, moment: pd.Timestamp):
        left = self.start <= moment if self.closed[0] else self.start < moment
        right = moment <= self.end if self.closed[1] else moment < self.end
        return (left and right)

    def __repr__(self):
        res = f"{self.start.isoformat()} - {self.end.isoformat()} ({self.timedelta.__str__()})"
        return res

    def __str__(self):
        return self.__repr__()

    def intersection(self, other):
        """not implemented yet"""
        return None

    def difference(self, other):
        """not implemented yet"""
        return None

    def union(self, other):
        """not implemented yet
        almost like .merge() below however self.closed must be considered
        (in .merge() it is not, always both ends closed)
        """
        return None

    def merge(self, other, fix="up"):
        """
        fix="up"  means new TimeInterval ends at max(self.end, other.end)
        fix="down"  means new TimeInterval begins at min(self.start, other.start)
        In both cases length of new TimeInterval is equal to the length of union of both TimeIntervals
        as in case of union of two intervals on the real line;
        i.e. if TimeIntervals overlap then theirs common part is condidered only once.
        """
        t_max = max(self.end, other.end)
        t_min = min(self.start, other.start)
        if self.end <= other.start or self.start >= other.end:
            timespan = self.timedelta + other.timedelta
        else:
            timespan = t_max - t_min
        if fix == "up":
            res = TimeInterval(t_max, -timespan)
        elif fix == "down":
            res = TimeInterval(t_min, timespan)
        return res

    def __add__(self, other):
        return self.merge(other)


# %%
def is_date_around_list(
        date: pd.Timestamp,
        dates_list: pd.Series,
        timedelta: pd.Timedelta,
        closed: Tuple[bool, bool] = (True, True), ) -> bool:
    """
    Checks if `date` is near any element of the `dates_list`
    where 'near' means that for any `d` in `dates_list`
     `date <= d <= date + timedelta`  for  `timedelta` positive
    or
     `date + timedelta <= d <= date`  for  `timedelta` negative.
    `<=` sign is replaced with `<` if respective entry of `closed` is False (see below).
    Arguments
    ---------
    date : pd.Timestamp
    dates_list : List[pd.Timestamp]
    timedelta : pd.Timedelta
        if _positive_ then we check if the `date` precedes some `d` from `dates_list`
         by no more then `timedelta`
        if _negative_ then we check if the `date` follows some `d` from `dates_list`
         by no more then `timedelta`
    closed : (bool, bool) : (True, True);
        consider relative ends of the `interval = sorted(date, data + timedelta)`
        as belonging to it (True) or not (False);
        i.e. 'True' is equiv. to '[' or ']' and 'False' to '(' or ')' in math notation.
    """
    res = any(dates_list.apply(lambda d: d in TimeInterval(date, timedelta, closed)))
    return res

# %%
