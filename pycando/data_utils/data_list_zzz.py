#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: DataList, DataItem
version: 1.0
type: module
keywords: [list of data objects, ...]
description: |
    Class for collecting data objects of the same type.
    These objects may be 'simple' data frames or arrays or lists, but may be also any other class.
    May come from many files or be parts of one file or be created on the fly by some common procedure.
    The assumption is all these objects have the same structure as deep as possible.
    Ideally only data values (and size) are different.
    Most important features of DataList are:
    - pretty printing with use of builtin.printing.indent();
    - len(), .N
    - slicing and selecting like other Python's collections
    - .loc, .iloc, .nam (changing keys to items names ~= .loc but .loc is restricted here for item ID != item name);
    -
file:
    date: 2023-04-15
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
from dataclasses import dataclass  # , field
import abc
from copy import deepcopy
from pathlib import Path

import pandas as pd
import darts as ds

import utils.adhoc as ah
import utils.builtin as bi
import utils.df as udf
from utils.project import DataSpec, Paths


# %%
@bi.repr
@dataclass
class DataItem:
    index: int      # must be -- index of a data file (from DataSpec)
    name: str = None        # ~= real site name & data portion (from file name _stem_)
    #
    file_prep: str = None
    ts: ds.TimeSeries = None
    df: pd.DataFrame = None
    err: str = None


@bi.repr
class DataList(abc.ABC):

    def __init__(
        self,
        items: dict = None,
        **kwargs,
    ):
        """"""
        self.items = bi.coalesce(items, dict())

    # %% loading data into object

    @abc.abstractmethod
    def load_raw(self, silent=False, *args, **kwargs) -> bool:
        """loading from raw data (usually .csv)
        silent: bool = False
            do print info on loading advances and issues?
        """
        return True

    @abc.abstractmethod
    def load(self, silent=False, *args, **kwargs) -> bool:
        """loading from prepared data (usually .pkl)
        silent: bool = False
            do print info on loading advances and issues?
        """
        return True


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
            new = cls(**data)
            return new
        except Exception as e:
            print(e)
            return False

    # %%

    def __len__(self) -> int:
        return len(self.items)

    @property
    def N(self) -> int:
        """alias for len(self)"""
        return len(self.items)

    @property
    def keys(self) -> list:
        """
        Keys of `.items`.
        it's @property (although it's usually function) as .indices (below) is
        """
        return self.items.keys()

    @property
    def indices(self) -> list:  # dict_keys
        """
        `.index` entries of `.items`.
        It's the same as `self.keys` iff `self.loc_on == True`,
        thus always `self.loc.keys == self.indices`;
        """
        return [item.index for item in self.items.values()]

    @property
    def names(self) -> list:
        """
        `.name` entries of `.items`.
        #It's the same as `self.keys` iff `self.loc_on == True`,
        thus always `self.nam.keys == self.names`;
        """
        return [item.name for item in self.items.values()]

    # %% getting elements

    def __only_site(self, dic: dict, site: str = None) -> dict:
        """"""
        if site is not None:
            dic = {k: path for k, path in dic.items() if site == str(Path(path).parent)}
        return dic

    def __update_files_dicts(self) -> None:
        self.files_prep = {k: f for k, f in self.files_prep.items() if k in self.indices}
        self.files_raw = {k: f for k, f in self.files_raw.items() if k in self.indices}

    @property
    def loc_on(self) -> bool:
        """
        If True then keys of .items are equal to respective "index" values (of each item entry);
        if False then keys are simply consecutive numbers 0, 1, ..., .self.N - 1
        (which may be not ordered if e.g. `.sort(retain_keys=True)` was aaplied).
        """
        loc_on = list(self.items.keys()) == [v.index for v in self.items.values()]
        return loc_on

    @property
    def loc(self):
        """
        returns `self` with key of each item set to it's "index" entry;
        """
        new = self.copy
        if not new.loc_on:
            new.items = {v.index: v for k, v in new.items.items()}
        return new

    @property
    def nam_on(self):
        """
        If True then keys of .items are equal to respective "name" values (of each item entry);
        if False then keys are simply consecutive numbers 0, 1, ..., .self.N - 1
        (which may be not ordered if e.g. `.sort(retain_keys=True)` was aaplied).
        """
        nam_on = list(self.items.keys()) == [v.name for v in self.items.values()]
        return nam_on

    @property
    def nam(self):
        """
        returns `self` with key of each item set to it's "name" entry;
        !!! this may not work with slices (not tested at all) !!!
        but convenient in some summaries, like  .info()
        """
        new = self.copy
        if not new.nam_on:
            new.items = {v.name: v for k, v in new.items.items()}
        return new

    def enumerate(self):
        """Sets new keys for .items 0, 1, ..., self.N - 1."""
        new = self.copy
        new.items = {i: v for i, v in enumerate(new.items.values())}
        return new

    def __slice(self, keys):
        """"""
        keys = list(keys) if not isinstance(keys, (list, tuple, range, slice)) else keys
        #
        new = self.copy
        # lookup series
        idx = pd.Series(range(new.N))
        idx.index = new.keys  # may be range(self.N) but rearranged or .index, or...
        # select .items
        if isinstance(keys, slice):
            if new.loc_on:
                new.items = {k: v for k, v in new.items.items()
                               if k in idx.loc[keys].index}
            else:
                new.items = {k: v for k, v in new.items.items()
                               if k in idx[keys].index}
        else:
            if isinstance(keys[0], bool):
                new.items = {k: v for k, v in new.items.items() if k in idx[keys].index}
            else:
                new.items = {k: v for k, v in new.items.items() if k in keys}
        #
        new.__update_files_dicts()
        return new

    def __getitem__(self, keys):
        """
        dl = DataList(...)
        k is int
        dl[k] -- k-th
        """
        keys = list(self.keys) if keys is None else keys
        if isinstance(keys, int):
            # just one element of self.items
            return self.items[keys]
        else:
            # for keys Iterable get relevant part of the whole structure
            return self.__slice(keys)

    # %%

    @property
    def sites(self) -> list:
        """list of site names taken from parent name of each item's .file entry
        """
        sites = [item.site for item in self.items.values()]
        return sites

    @property
    def sites_dict(self) -> dict:
        """
        dictionary {key: site_name} for all items,
        where site names are taken from parent name of each item's .file entry
        """
        sites = {k: item.site for k, item in self.items.items()}
        return sites

    def only_site(self, site: str):
        """
        Returns copy of DataList having only items relevant to given site.
        item is retained in this copy iff
        `site` is contained (in a sense of string) == a _parent_ directory of an item.file;
        Do not confuse it with an item.name entry which is a _stem_ (name without extension) of an item.file.
        """
        new = self.copy
        new.site = site
        new.files_raw, new.files_prep = (new.__only_site(dic, site) for dic in (new.files_raw, new.files_prep))
        new.items = {k: item for k, item in new.items.items() if site == item.site}
        new = new.enumerate()
        return new

    # %%  stats and summaries

    def errors(self):
        """prints info on all errors occurred during loading data from files (via .load() or .load_raw())
        """
        for f, item in self.items.items():
            if item.err:
                print(f"{f} : {item.file}")
                print(item.err)

    def _apply_summary(
        self, summary,
        df: bool = False, T: bool = False, columns: list[str] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        """
        Apply `summary` on data (.df or .ts entry of an item)
        to each item of DataList and concatenates the result into one data frame;
        concatenatian is always row-wise i.e. along 0-axis:
            pd.concat({k0: sum0, ..., kN: sumN}, axis=0);
        df: bool = False,
            if False then .ts component of each item is summarised; otherwise .df is taken;
        T: bool = False,
            do transpose the result for each item before final concatenation?
        columns: list[str] = None,
            take only these columns (variables, or 'components' in darts.TimeSeries);
            if None, then takes all columns;
        *args, **kwargs
            passed to `summary`.
        """
        def get_frame(item):
            frame = item.df if df else item.ts.pd_dataframe()
            frame = frame[columns] if columns else frame
            return frame
        res = {k: summary(get_frame(item), *args, **kwargs) for k, item in self.items.items()}
        if T:
            res = {k: info.T for k, info in res.items()}
        res = pd.concat(res)
        return res

    def summary(
        self,
        df: bool = False, T: bool = False, columns: list[str] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        """summary on data for all items;
        summary is obtained via utils.df.summary();
        df: bool = False,
            if False then .ts component of each item is summarised; otherwise .df is taken;
        T: bool = False,
            do transpose the result for each item before final concatenation?
        columns: list[str] = None,
            take only these columns (variables, or 'components' in darts.TimeSeries);
            if None, then takes all columns;
        *args, **kwargs
            passed to utils.df.summary().
       """
        res = self._apply_summary(udf.summary, df, T, columns, *args, **kwargs)
        return res

    def summary_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        alias for self.summary(df=True, ...)
        """
        return self.summary(df=True, *args, **kwargs)

    def info(
        self,
        df: bool = False, T: bool = False, columns: list[str] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        """
        The same as .summary() but uses utils.df.info()
        which is in fact the same as utils.df.summary() but with different defaults.
        """
        res = self._apply_summary(udf.info, df, T, columns, *args, **kwargs)
        return res

    def info_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        alias for self.info(df=True, ...)
        """
        return self.info(df=True, *args, **kwargs)

    # %%

    def __sort_items_list(self, items_list: list) -> list:
        """sorts list of DataItems
        not used
        """
        items_list = sorted(items_list, key=lambda item: item.ts.time_index[0])
        return items_list

    def __sort_ts_list(self, ts_list: list) -> list:
        """sorts list of TimeSeries according to first element of theirs time_index.
        """
        ts_list = sorted(ts_list, key=lambda ts: ts.time_index[0])
        return ts_list

    def merge_ts(self, site: str = None) -> pd.DataFrame:
        """
        Returns all darts.TimeSeries relative to given `site`
        concatenated along time axis into one TimeSeries.
        `site` must be given directly or be specified earlier for a DataList (during initialisation).
        """
        if site is None:
            if self.site is None:
                raise Exception("There is no `site` which must be specified either directly or in this DataList.")
            else:
                items = self.items
        else:
            if self.site is None:
                items = self.only_site(site).items
            elif site != self.site:
                raise Exception(f"This DataList has only data for other site = {self.site}.")
            else:
                items = self.items
        ts_list = self.__sort_ts_list([item.ts for item in items.values()])
        ts = ds.concatenate(ts_list)
        return ts

    def merge_df(self, site: str = None, freq: str = '15T', fill_value=None, dropna: bool = True) -> pd.DataFrame:
        """
        Returns all pd.DataFrame relative to given `site`
        concatenated along axis 0 into one DataFrame.
        `site` must be given directly or be specified earlier for a DataList (during initialisation).
        Notice that `dropna = True` results only in dropping leading and trailing NaNs records,
        as all the discontinuities in time (wrt. `freq`) are filled with `fill_value` (which is None by default).
        """
        if site is None:
            if self.site is None:
                raise Exception("There is no `site` which must be specified either directly or in this DataList.")
            else:
                df = pd.concat(item.df for item in self.items.values())
        else:
            if self.site is None:
                items = self.only_site(site).items
                df = pd.concat(item.df for item in items.values())
            elif site != self.site:
                raise Exception(f"This DataList has only data for other site = {self.site}.")
            else:
                df = pd.concat(item.df for item in self.items.values())

        df.index = df.time
        if dropna:
            df = df.dropna()
        freq = pd.infer_freq(df.index) if freq is None else freq
        df = df.asfreq(freq, fill_value)
        return df

    @property
    def ts(self):
        """
        Returns all darts.TimeSeries in this DataList concatenated along time axis into one TimeSeries.
        However it works  iff  this DataList has `site` entry specified
        (i.e. was initialised with some value passed to `site` arg).
        This is shorthand for ("property" version of) .merge_ts() method
        (which has only `site` arg. taken here from self.).
        """
        return self.merge_ts()

    @property
    def df(self):
        """
        Returns all pd.DataFrames in this DataList concatenated along axis 0 into one DataFrame.
        However it works  iff  this DataList has `site` entry specified
        (i.e. was initialised with some value passed to `site` arg).
        This is shorthand for ("property" version of) .merge_df() with all the args set to default.
        """
        return self.merge_df()

# %%
