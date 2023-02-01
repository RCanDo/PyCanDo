#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: date/time helpers - events
version: 1.0
type: module             # module, analysis, model, tutorial, help, example, ...
keywords: [categorical time series, datetime, aggregation, events]
description: |
remarks:
todo:
    - perhaps separate class for each window would be better
      but it poses number of tech. problems
      while simple dictionary serves very well;
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
from typing import Union, Tuple, Iterable, Dict
from copy import deepcopy

# import numpy as np
import pandas as pd

from .ts import TimeInterval
import common.builtin as bi


# %% what happend before all the false alarms
# HEAVY STUFF !!!

class WindowsAtMoments():
    """
    class for gathering data on events around (usually before) some fixed moments;
    """

    def __init__(
            self,
            timedelta: Union[pd.Timedelta, str, int],  # windows span along time axis; negative for looking back
            moments: Union[Iterable[Union[str, pd.Timestamp]], pd.DatetimeIndex, pd.Index],
            events: Union[Iterable[Union[str, pd.Timestamp]], pd.DatetimeIndex, pd.Index],
            offset: Union[pd.Timedelta, str, int] = 0,
            #   # as for timedelta: measured along time axis; details below (important!)
            closed: Tuple[bool, bool] = (True, True),    # respective endponint belongs or not to the window
            process: bool = True,
            #   # False when creating instance from `data` i.e. only copying them into the structure
            *args, **kwargs, ):
        """see ._gather_all() description
        """
        if isinstance(timedelta, str):
            timedelta = pd.Timedelta(timedelta)

        if isinstance(offset, (str, int)):
            offset = pd.Timedelta(offset)

        self.closed = closed
        self.moments = pd.to_datetime(moments)     # idempotent - no worry!
        self.events = pd.to_datetime(events)

        self.timedelta = timedelta
        self.offset = offset

        if process:
            self.windows = \
                self._gather_all(self.timedelta, self.moments, self.events, self.offset, self.closed)
        else:
            # for init from "data", see  cls.from_data()
            self.windows = kwargs.get('windows', dict())

        # self.loc_on = list(self.windows.keys()) == [v["index"] for v in self.windows.values()]

    def _gather_all(
            self,
            timedelta: pd.Timedelta,  # windows span along time axis; negative for looking back
            moments: Union[Iterable[pd.Timestamp], pd.DatetimeIndex, pd.Index],
            events: Union[Iterable[pd.Timestamp], pd.DatetimeIndex, pd.Index],
            offset: pd.Timedelta,  # as for timedelta: measured along time axis; details below (important!)
            closed=(True, True), ) -> Dict[int, Tuple[pd.Timestamp, int, pd.Series]]:
        #   # respective endponint belongs or not to the window
        """
        `moments` and `events` encode datetime data
        via str, or proper datetime format;
        nevertheless they are always turned to datetime;
        internally they are turned to  pd.Series  with  datetime values  and with  integer  index  !!!

        Gathers  moments  of all events (moments in `events`)
        which happend during fixed length of time: `timedelta`,
        within each window around each moment from a list of selected moments: `moments`;

        `offset` shifts each moment `m` in `moments` along the time axis
        (i.e. if negative then all `m`s will be shifted backwards).

        It means that for given `t` (`timedelta`) and `o` (`offset`)
        and for all `m` in `moments`
        we gather all events (data points) from `events`
        in the following `window` (time interval)
            [m + o, m + o + t]  for t > 0,
            [m + o + t, m + o]  for t < 0;
        programatically `window` is constructed by
            ti = TimeInterval(m + o, t, closed)
        where always  ti.start <= ti.end  i.e. `m + o + t` and `m + o` are sorted.
        `closed` is tuple of two booleans informing if the respective endpoint belongs to interval.
        E.g. if `closed = (True, False)` then we get window
            [m + o, m + o + t)  for t > 0,
            [m + o + t, m + o)  for t < 0;
        For each `m` the procedure also gathers moments falling into
            (m, m + o)  for o > 0,
            (m + o, m)  for o < 0;
        these events are called `offsets` and the interval vanishes for `o = 0` (default).
        The 'offsets' interval is always open at `m` point
        while the point `m + o` belongs only to one of the `window` or `offsets`
        and it is inferred from values of `o`, `t` and `closed`
        (quite a number of cases to consider).
        I.e. events at this point always fall into only one of `window` or `offsets`.

        Finally events _at_ `m` are also gathered separately under data entry `at`.
        If `o = 0` and 'window' is closed at `m` then events at this point
        will be gathered twice: as part of `window` and as `at`.

        Notice that if `o > 0` and `t < 0` then `window` and `offsets` overlap
        even more: one is contained in the other (except `o + m` point).
        Obviously the same holds if `o < 0` and `t > 0`.

        Returns dictionary:
        windows = {k: {"moment": moment_k,  # k-th moment from `moments` (pd.Timestamp)
                       "index": index_k,    # is value of index of k-th `moments` (int)
                       "window": window_k_events,   # subseries of `events` which happend during k-th `window`
                       "offsets": offset_k_events,  # subseries of `events` which happend during k-th `offset`
                       "at": moment_k_events        # subseries of `events` which happend at k-th `moment`
                       }
                   for  k  in range(len(`moments`))
                   }
        """
        windows = dict()

        if offset.value > 0 and timedelta.value > 0:
            closed_offset = (False, not closed[0])
        elif offset.value > 0 and timedelta.value < 0:
            closed_offset = (False, not closed[1])
        elif offset.value < 0 and timedelta.value > 0:
            closed_offset = (not closed[0], False)
        elif offset.value < 0 and timedelta.value < 0:
            closed_offset = (not closed[1], False)
        else:
            closed_offset = (False, False)
            # i.e. nothing for offset == 0 (default)
            # pro-forma as it's not used anywhere in this case

        for k in range(len(moments)):
            print(k)    # for knowing the progress in terms of moments length
            idx_k = moments.index[k]   # k-th .index
            moment_k = moments.iloc[k]   # k-th .value
            print(moment_k)
            #
            window_k = TimeInterval(moment_k + offset, timedelta, closed)
            window_k_events = events.apply(lambda e: e in window_k)
            window_k_events = events[window_k_events]
            print(f" window: {len(window_k_events)}")
            #
            if offset.value == 0:
                offset_k_events = pd.Series()
            else:
                offset_k = TimeInterval(moment_k, offset, closed_offset)
                offset_k_events = events.apply(lambda d: d in offset_k)
                offset_k_events = events[offset_k_events]
            print(f"offsets: {len(offset_k_events)}")
            #
            at_k_events = events[events == moment_k]
            print(f"     at: {len(at_k_events)}")
            #
            windows[k] = {
                "moment": moment_k,
                "index": idx_k,
                "window": window_k_events,
                "offsets": offset_k_events,
                "at": at_k_events, }

        return windows

    # %% representation
    def __str__(self) -> str:
        return bi.indent(self)

    def __repr__(self) -> str:
        return bi.indent(self)

    def print(self, *args, **kwargs) -> None:
        print(bi.indent(self, *args, **kwargs))

    # %% copy and pure data
    @property
    def copy(self):
        return deepcopy(self)

    @property
    def data(self) -> dict:
        """returns self as dictionary but only data i.e. without methods (callables)"""
        res = {k: v for k, v in self.__dict__.items() if not callable(v)}
        return res

    @classmethod
    def from_data(cls, data: dict):
        """just copying data into structure"""
        try:
            new = cls(process=False, **data)
            return new
        except Exception as e:
            print(e)
            return False

    # %%
    def __add__(self, other):
        """merging two WindowsAtMoments objects"""
        if self.timedelta != other.timedelta:
            raise Exception('cannot merge: different .timedelta')
        if self.offset != other.offset:
            raise Exception('cannot merge: different .offset')
        if self.closed != other.closed:
            raise Exception('cannot merge: different .closed')
        if any(self.events != other.events):
            raise Exception('cannot merge: different .events')
        new = self.loc.copy()   # keys == indices
        other = other.loc
        new.moments = pd.concat([self.moments, other.moments]).drop_duplicates()
        new.windows = {**new.windows, **other.windows}  # keys are always unique, the second is taken
        new = new.sort(by="moments")
        return new

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def N(self) -> int:
        """alias for len(self)"""
        return len(self.windows)

    @property
    def loc_on(self) -> bool:
        """
        If True then keys of .windows are equal to respective "index" values (of each windows entry);
        if False then keys are simply consecutive numbers 0, 1, ..., .self.N - 1
        (which may be not ordered if e.g. `.sort(retain_keys=True)` was aaplied).
        """
        loc_on = list(self.windows.keys()) == [v["index"] for v in self.windows.values()]
        return loc_on

    @property
    def no_offset(self) -> bool:
        """checks for 0 offset"""
        return self.offset.value == 0

    def recount(self):
        """
        recalculation of everything;
        returns new object;
        """
        new = WindowsAtMoments(process=True, **self.data)
        return new

    def enumerate(self):
        """Sets new keys for .windows 0, 1, ..., self.N - 1."""
        new = self.copy
        new.windows = {i: v for i, v in enumerate(new.windows.values())}
        return new

    def sort(self, by: str = "moments", ascending: bool = True, retain_keys: bool = False):
        """
        Sorting `.windows` wrt to `.index` (int) or `.moment` (pd.Timestamp) entry.
        By default it does not retain keys of `.windows` (`retain_keys = False`)
        i.e. after sorting `.windows` are enumerated anew, from 0 to self.N - 1.
        If `retin_keys = True` then old key for each `.windows` entry is retained.
        """
        if retain_keys:
            # lookup series
            keys = pd.Series(self.keys)
            keys.index = self.index
        #
        new = self.loc
        # sort .moments
        if by == "moments":
            new.moments = new.moments.sort_values(ascending=ascending)
        elif by == "index":
            new.moments = new.moments.sort_index(ascending=ascending)
        else:
            raise Exception("`by` must be 'moments' or 'index'.")
        # sort .windows
        new.windows = {k: new.windows[k] for k in new.moments.index}
        #
        if retain_keys:
            new.windows = {keys[k]: v for k, v in new.windows.items()}
        else:
            new = new.enumerate()
        return new

    @property
    def loc(self):
        """
        returns `self` with key of each window set to it's "index" entry;
        """
        new = self.copy
        if not new.loc_on:
            new.windows = {v['index']: v for k, v in new.windows.items()}
        return new

    @property
    def keys(self):
        """
        Keys of `.windows`.
        it's @property (although it's usually function) as .index (below) is
        """
        return self.windows.keys()

    @property
    def index(self) -> pd.Index:
        """
        `.index` entries of `.windows`.
        It's the same as `self.keys` iff `self.loc_on == True`,
        thus always `self.loc.keys == self.index`;
        """
        return self.moments.index

    def __slice(self, keys):
        """"""
        keys = list(keys) if not isinstance(keys, (list, tuple, range, slice)) else keys
        #
        new = self.copy
        # lookup series
        idx = pd.Series(range(new.N))
        idx.index = new.keys  # may be range(self.N) but rearranged or .index, or...
        # select .windows
        if isinstance(keys, slice):
            if new.loc_on:
                new.windows = {k: v for k, v in new.windows.items()
                               if k in idx.loc[keys].index}
            else:
                new.windows = {k: v for k, v in new.windows.items()
                               if k in idx[keys].index}
        else:
            if isinstance(keys[0], bool):
                new.windows = {k: v for k, v in new.windows.items() if k in idx[keys].index}
            else:
                new.windows = {k: v for k, v in new.windows.items() if k in keys}
        # select .moments
        try:
            new.moments = new.moments.iloc[keys]
        except IndexError:
            new.moments = new.moments[keys]
        #
        return new

    def __getitem__(self, keys):
        """
        eb = WindowsAtMoments(...)
        k is int
        eb[k] -- kt-th
        """
        keys = list(self.keys) if keys is None else keys
        if isinstance(keys, int):
            # just one element of self.events_before_each
            return self.windows[keys]
        else:
            # for keys Iterable get relevant part of the whole structure
            return self.__slice(keys)

    def head(self, h=3):
        """first h windows of `self` returned as new WindowsAtMoments
        """
        keys = list(self.keys)[:h]
        return self.__slice(keys)

    def _union_index_k(
            self, k: int,
            what: Iterable[str] = ('all',), ) -> Union[pd.Index, pd.Series]:
        """"""
        idx = pd.Index([])
        for w in what:
            idx = idx.union(self[k][w].index)
        return idx

    def _what(self, what: Union[str, Iterable[str]]) -> Iterable[str]:
        """"""
        if isinstance(what, str):
            what = (what,)
        what = list(what)
        if 'all' in what:
            if self.no_offset:
                what = ['window', 'at']
            else:
                what = ['window', 'offsets', 'at']
        return what

    def union(
            self,
            what: Union[str, Iterable[str]] = ('all',),
            events: bool = False, ) -> Union[pd.Index, pd.Series]:
        """
        union of indices (or respective events) across all windows;
        and `what`s i.e. entries of each window;
        possible values for `what` are 'all', 'window', 'offsets', 'at' and any combination of them;
        (compare relevant keys in self[k] == self.windows[k])
        if `events` False, index will be returned (default);
        if `events` True, series of relevant events (date-times) will be returned.
        """
        what = self._what(what)
        #
        result = pd.Index([])
        for k in self.keys:
            result = result.union(self._union_index_k(k, what))
        if events:
            result = self.events[result]
        return result

    def complement(
            self,
            what: Union[str, Iterable[str]] = ('all',),
            events: bool = False, ) -> Union[pd.Index, pd.Series]:
        """
        series complementary to relevant self.union(what, events) wrt. self.events
        """
        result = self.union(what, events=False)
        result = self.events.index.difference(result)
        if events:
            result = self.events[result]
        return result

    def timespan(
            self,
            what: Union[str, Iterable[str]] = None, ) -> pd.Timedelta:
        """
        !!! object must be first sorted by 'moments' in ascending order to get proper result !!!
        but we do not sort it by default internally as it's expensive and may be already done;
        it's up to the user
        """
        if what is None:

            def timeinterval(k):
                moment_k = self[k]['moment']
                window_k = TimeInterval(moment_k + self.offset, self.timedelta)
                start_k = min(moment_k, window_k.start)
                end_k = max(moment_k, window_k.end)
                return TimeInterval(start_k, end_k)
            intervals = [timeinterval(k) for k in self.keys]
        else:
            what = self._what(what)

            def timeinterval(k):
                events_k = sum((self[k][w].tolist() for w in what), [])
                start_k = min(events_k)
                end_k = max(events_k)
                return TimeInterval(start_k, end_k)
            intervals = [timeinterval(k) for k in self.keys]
        # res = intervals[0]
        # for ti in intervals:
        #     res = res.merge(ti)
        res = sum(intervals, intervals[0])
        return res.timedelta

    def info(self) -> None:
        print(f"    timedelta : {self.timedelta}")
        print(f"       offset : {self.offset}")
        print(f"       closed : {self.closed}")
        print(f"       length : {self.N}")
        print(f"nr of moments : {len(self.moments)}")
        print(f"nr of events  : {len(self.events)}")

    @property
    def stat(self) -> pd.DataFrame:
        """"""
        res = {k: {'index': v['index'],
                   'moment': v['moment'],
                   'window': len(v['window']),
                   'offsets': len(v['offsets']),
                   'at': len(v['at']),
                   }
               for k, v in self.windows.items()}
        res = pd.DataFrame.from_dict(res, orient='index')
        res['elapsed'] = res['moment'].diff()
        res = res[['index', 'moment', 'elapsed', 'window', 'offsets', 'at']]
        return res

    @property
    def stats(self) -> pd.DataFrame:
        """alias for self.stat"""
        return self.stat

    # %%
    def counts_by_windows(
            self,
            events: pd.DataFrame,
            what: Union[str, Iterable[str]] = ("window",),
            index: bool = True, ) -> pd.DataFrame:
        """
        Returns counts of each level of each factor of `events`
        within each window of `self`.
        (`events` here is not `self.events` but original df from which `self.events` was taken.)

        It is assumed that:
        - `self` should be derived first from `events` (i.e. `self.events` comes from `events` df);
        - all columns of `events` are categorical (factors)
          (thus the column which is origin of `self.events` should be removed).

        Result columns are 2-level MultiIndex where the first level
        gives a name of a variable and the second level gives all
        levels of the respective variable (factor!).

        Indices are ids of windows if `index` is False, thus they are in `renge(self.N)`.
        If `index` is True (default) then indices are respective `index` entry
        of each window i.e. original index of the `moment` (in `events` df)
        for which data are gathered in the window.

        Values are counts for each factor level within each window.

        `what` works the same way as in `self.union()`.

        See common.df.helpers.count_factors_levels() as a simpler version of it
        (summarising one data frame and returning one row).
        """
        what = self._what(what)
        cols = events.columns
        frames_before = {v: pd.DataFrame() for v in cols}
        for k in self.keys:
            events_k = events.loc[self._union_index_k(k, what), :]
            for v in cols:
                vc = events_k[v].value_counts()
                if index:
                    vc.name = self[k]['index']
                else:
                    vc.name = k
                frames_before[v] = pd.concat([frames_before[v], vc], axis=1)

        for v in frames_before.keys():
            print(f"{v.rjust(15)} : {frames_before[v].shape}")
        # e.g.
        #         State : (26, 89)
        #     StateInfo : (10, 89)
        # CommandSource : (11, 89)
        #       Element :  (9, 89)
        #   ElementInfo :  (5, 89)

        frame_before = pd.concat(frames_before).T
        frame_before.fillna(0, inplace=True)

        for v in cols:
            frame_before[v] = frame_before[v].astype(int)

        return frame_before

    def clusters_data(self, distance=None):
        """
        returns summary like
                   clusters  in-clusters  all
        non-empty        12          260  272
        empty           143            2  145
        all             155          262  417
        where
        - clusters (nr of clusters == nr of alarms for which which at least 5 min 30 sec elapsed from previous alarm)
        - in-clusters (opposite of the above == all alarms which are non-first in clusters)
        - empty (nothing happend in the window)
        - non-empty (sth happend in the window)
        """
        if distance is None:
            distance = -(self.timedelta + self.offset)
        else:
            distance = pd.Timedelta(distance)

        idx_empty = (self.loc.stat.window == 0)

        # inside clusters (NOT first in cluster)
        idx_in = (self.loc.stat.elapsed <= distance)

        idx_in.name = 'clusters'
        idx_empty.name = 'empty'

        df = pd.crosstab(idx_empty, idx_in, margins='all')

        # sometimes df collapses to one row or one column
        col_ren = {False: "clusters", True: "in-clusters", "All": "sum"}
        row_ren = {False: "non-empty", True: "empty", "All": "sum"}

        df.columns = [col_ren[c] for c in df.columns]
        df.index = [row_ren[c] for c in df.index]

        return df, idx_empty, idx_in

    def clusters_table(self, distance=None):
        """"""
        df, idx_empty, idx_in = self.clusters_data(distance)
        return df


# %% alias (which name is better?)
WindowsAround = WindowsAtMoments

# %%
