#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

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
    usage:
        interactive: False  # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2022-10-01
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
# from functools import partial
from typing import Any
# import numpy as np
import pandas as pd

from .builtin import lengthen


# %%
# %%
def section(title: str):
    print("\n")
    print(title)
    print("-" * len(title))


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


def indent(obj: Any, ind: str = "   ", level: int = 0, head: int = 0, mx: int = 120, info=True) -> str:
    """
    returns str
    """
    head = lengthen(head, 2)  # second entry for nr of columns in df

    def type_as_str(obj):
        res = str(type(obj)).strip("<>").strip("class ").strip("'")
        return res

    def repr_data_frame(df):
        info = f"<pd.DataFrame of shape  {df.shape[0]}, {df.shape[1]}>"
        if head[0] == 0:
            res = info
        else:
            df = df.iloc[:head[0], :head[1]].copy()
            if info:
                res = info + "\n" + f"{df!r}"
            else:
                res = "\n" + f"{df!r}"
        return res

    def repr_series(series):
        info = f"<pd.Series of length  {len(series)}>"
        if head[0] == 0:
            res = info
        else:
            series = series.iloc[:head[0]].copy()
            if info:
                res = info + "\n" + f"{series!r}"
            else:
                res = "\n" + f"{series!r}"
        return res

    def res_list_tuple(obj):
        info = f"<{type_as_str(obj)} of length  {len(obj)}>"
        if head[0] == 0:
            res = info
        else:
            obj = obj[:head[0]]
            if info:
                res = info + "\n" + f"{obj!r}"
            else:
                res = f"{obj!r}"
        return res

    def res_set(obj):
        info = f"<{type_as_str(obj)} of length  {len(obj)}>"
        if head[0] == 0:
            res = info
        else:
            obj = set(list(obj)[:head[0]]) if len(obj) > head[0] else obj
            if info:
                res = info + "\n" + f"{obj!r}"
            else:
                res = f"{obj!r}"
        return res

    def res_object(obj):
        """not used"""
        try:
            length = len(obj)
        except TypeError:
            length = len(obj.__dict__)
        #
        info = f"<{type_as_str(obj)} of length  {length}>"
        if head[0] == 0:
            res = info
        else:
            if info:
                res = info + "\n" + f"{obj!r}"
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
            res = "\n" + ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, pd.Series):
        res = repr_series(obj)
        if head[0] > 0:
            res = "\n" + ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, (list, tuple)):
        res = res_list_tuple(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, set):
        res = res_set(obj)
        if head[0] > 0:
            res = ("\n" + ind * level).join(s for s in res.split("\n"))
    elif isinstance(obj, str):
        res = f"{obj!r}"
    elif isinstance(obj, (pd.Timedelta, pd.Timestamp)):
        res = f"{obj!r}"
    elif callable(obj):
        res = f"{obj!r}"
    else:     # what it could be ???
        try:
            res = "\n" + \
                "\n".join(ind * level + f"{k} : {indent(v, ind, level + 1, head, mx)}"
                          for k, v in obj.__dict__.items())
            # res = res_object(obj)
        except AttributeError:
            res = f"{obj!r}"
            res = res if len(res) <= mx else res[:mx] + "..."
    return res


def show(obj, ind="   ", level=0, head=0, mx=120):
    """prints with use of indent()"""
    res = indent(obj, ind, level, head, mx)
    return print(res)


# %%
class Repr():
    """
    class to inherit from to get pretty representation;
    doesn't work with dataclass;

    class MyClass(Repr):
        ...
    """
    def __init__(self):
        pass

    # representation
    def __str__(self) -> str:
        return indent(self)     # , head=5)

    def __repr__(self) -> str:
        return indent(self)     # , head=5)

    def print(self, *args, **kwargs) -> None:
        print(indent(self, *args, **kwargs))


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
    cls.__str__ = indent    # partial(indent, head=5)
    cls.__repr__ = indent   # partial(indent, head=5)

    return cls


# %%
