#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: intelligent indenting facility for pretty printing
version: 1.0
type: module             # module, analysis, model, tutorial, help, example, ...
keywords: [pretty printing, string representation]
description: |
    Similar to pprint, but more versatile.
    Primary objective is to have ready to use printing facility
    for complicated data structures
    - json-like i.e. interlaced dict/list but
    with leaves being anything from callables to pd.DataFrames.
    For large leaves (like DataFrames) only short info is displayed
    or sth like .head(h) where h is a parameter.
    Indentation is also a parameter provided by the user ("   " by default).
remarks:
    - ready to use but at early stage of development.
todo:
    - intelligent info on leaves;
    - pd.DataFrames as list of pd.Series i.e. in general: what is the leaf?
file:
    date: 2022-10-01
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
import os
# from functools import partial
from typing import Any
# import numpy as np
import pandas as pd

from utils.builtin import lengthen, coalesce


# %%
def section(title: str, out: Any = True, vsep: int = 1) -> None:
    if out:  # i.e. print it out
        if vsep > 0:
            print("\n" * vsep, end="")
        print(title)
        print("-" * len(title))
    else:
        # we want to have nice sections in the code
        # even if we not always want to print them out
        pass


# %% representation
def not_too_long(obj: Any, mx: int = 33) -> bool:
    """
    Checks if the object is not larger then given limit `mx`.
    Uses .shape or len().
    Returns True if none of the above applies.
    """
    try:
        ret = max(obj.shape) < mx
    except (AttributeError, ValueError):
        try:
            ret = len(obj) < mx
        except TypeError:
            ret = True
    return ret


def indent(
        obj: Any,
        ind: str = "   ",
        level: int = 0,
        head: int | list[int] = 0,
        mx: int = 120,
        info: bool = True,
        inherit: bool = False,
) -> str:
    """
    Arguments
    ---------
    obj: Any,
        object of any type may be represented according to user wish;
    ind: str = "   ",
        form of indentation for printing `obj`s children;
    level: int = 0,
        at what level of indentation to begin printing `obj`;
        0 means it begins with no left margin.
    head: int | list[int] = 0,
        how many elements of a children (like pd.DataFrame, or iterable) to print out
    mx: int = 120,
        max string repr length to print; NOT used.
    info: bool = True,
        do print info on the class/type and length of an object (useful but makes more clutter).
    inherit: bool = False,   # if False then each Repr child will use it's own print params;
        if False then each Repr child will use it's own print params;
        otherwise it will inherit from its parent.

    Returns
    -------
    String representation of the object formatted for easy reading.
    """
    head = lengthen(head, 2)  # second entry for nr of columns in df or in darts.TimeSeries

    def type_as_str(obj):
        res = str(type(obj)).strip("<>").strip("class ").strip("'") if type(obj) else "<object of unknown class>"
        return res

    def repr_data_frame(df):
        if info:
            info_res = f"<pd.DataFrame of shape  {df.shape[0]}, {df.shape[1]}>"
        if head[0] == 0 and info:
            res = info_res
        else:
            df = df.iloc[:head[0], :head[1]].copy()
            if info:
                res = info_res + "\n" + f"{df!r}"
            else:
                res = "\n" + f"{df!r}"
        return res

    def repr_series(series):
        if info:
            info_res = f"<pd.Series of length  {len(series)}>"
        if head[0] == 0 and info:
            res = info_res
        else:
            series = series.iloc[:head[0]].copy()
            if info:
                res = info_res + "\n" + f"{series!r}"
            else:
                res = "\n" + f"{series!r}"
        return res

    def repr_darts_time_series(ts):
        if info:
            info_res = \
            f"<darts.TimeSeries of shape: time: {ts.n_timesteps}, columns: {ts.n_components}, samples: {ts.n_samples}>"
        if head[0] == 0 and info:
            res = info_res
        else:
            ts = ts.values()[:head[0], :head[1]].copy()
            if info:
                res = info_res + "\n" + f"{ts!r}"
            else:
                res = "\n" + f"{ts!r}"
        return res

    def repr_list_tuple(obj):
        if info:
            info_res = f"<{type_as_str(obj)} of length  {len(obj)}>"
        if head[0] == 0 and info:
            res = info_res
        else:
            obj = obj[:head[0]]
            if info:
                res = info_res + "\n" + f"{obj!r}"
            else:
                res = f"{obj!r}"
        return res

    def repr_set(obj):
        if info:
            info_res = f"<{type_as_str(obj)} of length  {len(obj)}>"
        if head[0] == 0 and info:
            res = info_res
        else:
            obj = set(list(obj)[:head[0]]) if len(obj) > head[0] else obj
            if info:
                res = info_res + "\n" + f"{obj!r}"
            else:
                res = f"{obj!r}"
        return res

    def repr_object(obj):
        """not used"""
        if info:
            try:
                length = len(obj)
            except TypeError:
                length = len(obj.__dict__)
            info_res = f"<{type_as_str(obj)} of length  {length}>"

        if head[0] == 0 and info:
            res = info_res

        else:
            if info:
                res = info_res + "\n" + f"{obj!r}"
            else:
                res = f"{obj!r}"

        return res

    if isinstance(obj, dict):
        if obj:
            res = "\n" + \
                "\n".join(ind * level + f"{k} : {indent(v, ind, level + 1, head, mx)}"
                          for k, v in obj.items())
        else:
            res = "\n" + "dict()"
    elif isinstance(obj, pd.DataFrame):
        res = repr_data_frame(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, pd.Series):
        res = repr_series(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif str(type(obj)) == "<class 'darts.timeseries.TimeSeries'>":
        res = repr_darts_time_series(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, (list, tuple)):
        res = repr_list_tuple(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, set):
        res = repr_set(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, (pd.Timedelta, pd.Timestamp)):
        res = f"{obj!r}"
    elif isinstance(obj, (os.PathLike)):
        res = f"{obj!r}"
    elif isinstance(obj, str):
        res = f"{obj!r}"
    elif callable(obj):
        res = f"callable {obj!r}"
    else:
        try:
            if 'print_params' in obj.__dict__:
                if inherit:
                    res = "\n" + \
                        "\n".join(ind * level + f"{k} : {indent(v, ind, level + 1, head, mx, info, True)}"
                                 for k, v in obj.__dict__.items())
                else:
                    res = "\n" + \
                        "\n".join(ind * level + repr_key_value(k, v, level + 1)
                                 for k, v in obj.__dict__.items())
            else:
                # res = f"{obj!r}"
                res = repr_object(obj)
                if head[0] > 0:
                    res = ("\n" + ind * level).join(s for s in res.split("\n"))
        except AttributeError:
        # try:
        #     res = f"{obj!r}"
        #     res = res if len(res) <= mx else res[:mx] + "..."
        # except:
            res = type_as_str(obj)

    return res


def show(obj, ind="   ", level=0, head=0, mx=120, info=True):
    """prints with use of indent()"""
    res = indent(obj, ind, level, head, mx, info)
    return print(res)


iprint = show   # alias


# %%
class PrintParams:
    def __init__(self, ind, level, head, mx, info):
        self.ind = ind
        self.level = level
        self.head = head
        self.mx = mx
        self.info = info

    def __repr__(self):
        ss = f"(ind = {self.ind}, level = {self.level}, head = {self.head}, mx = {self.mx}, info = {self.info})"
        return ss


class Repr():
    """
    class to inherit from to get pretty representation;
    doesn't work with dataclass;
    use as first parent (espacially in connection with abc.ABC)

    class MyClass(Repr, OtherClass):
        ...
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_print_params()

    def __str__(self) -> str:
        return indent(self, **self.print_params.__dict__)

    def __repr__(self) -> str:
        return indent(self, **self.print_params.__dict__)

    def print(
            self,
            ind: str = None,
            level: int = None,
            head: int = None,
            mx: int = None,
            info: str = None
    ) -> None:
        res = indent(
            self,
            ind=coalesce(ind, self.print_params.ind),
            level=coalesce(level, self.print_params.level),
            head=coalesce(head, self.print_params.head),
            mx=coalesce(mx, self.print_params.mx),
            info=coalesce(info, self.print_params.info),
        )
        print(res)

    def set_print_params(
            self,
            *,
            ind: str = "   ",
            level: int = 0,
            head: int = (3, 3),
            mx: int = 120,
            info: bool = True,
    ) -> None:
        self.print_params = PrintParams(
            ind=ind,
            level=level,
            head=head,
            mx=mx,
            info=info,
        )


# alias;
# semantically for creating empty container with pretty representation (via indent());
# not for inheriting from (like Repr)
PrettyEmpty = Repr


# %%
def repr(cls):
    """
    class decorator to give it pretty representation;
    works also with dataclass;

    @repr
    class MyClass():
        ...
    """

    def rprint(self, *args, **kwargs) -> None:
        print(indent(self, *args, **kwargs))

    cls.print = rprint
    cls.__str__ = indent    # partial(indent, , head=(3, 3))
    cls.__repr__ = indent   # partial(indent, , head=(3, 3))

    return cls


# %%
"""
empty = bi.Repr()
empty
empty.a = 1
empty.b = 'qq'
empty.sum = sum

empty
empty.sum([1, 2])
"""
# %%
