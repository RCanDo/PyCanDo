#! python3
# -*- coding: utf-8 -*-
"""
---
title: Conveniences And Utilities
subtitle: Based only on built-ins and basic libraries.
version: 1.0
type: module
keywords: [flatten, coalesce, ... ]
description: |
    Convenience functions and utilities used in everyday work.
remarks:
    - We use only basic packages from standard library
      like functools, itertools, typing, time, math
todo:
sources:
file:
    date: 2021-11-20
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

# %%
from pathlib import Path, PosixPath
from typing import Union, Any, Dict, Set, Iterable, List  # , Tuple, Optional, NewType,
# from __future__ import annotations

import time
# import common.builtin.timer as t

from functools import reduce, wraps
from itertools import zip_longest
import itertools as it
from copy import deepcopy

import math as m


# %%
def coalesce1(*args, empty: tuple[Any] = (None,)) -> Any | None:
    """
    depricated as coalesce() via loop is much faster;
    left for reference

    As in PostgreSQL: returns first not None argument;
    if all arguments are None then returns None.

    However it is possible to provide own empties via `empty` keyword-argument:
    each element of `empty` tuple is considered as being "empty" value
    and is ommited when searching for first non-empty.
    If all arguments are empty then first element of `empty` (None by default) is returned.
    """
    ll = list(filter(lambda x: x not in empty, args))
    ll.append(empty[0])     # in case ll is empty
    return ll[0]


def coalesce0(*args, empty: tuple[Any] = (None,)) -> Any | None:
    """
    As in PostgreSQL: returns first not None argument;
    if all arguments are None then returns None.

    However it is possible to provide own empties via `empty` keyword-argument:
    each element of `empty` tuple is considered as being "empty" value
    and is ommited when searching for first non-empty.
    If all arguments are empty then first element of `empty` (None by default) is returned.

    This is fast version
    where we use `==` operator for checking if element is in `empty`,
    i.e. we check `arg in empty` (for which `==` is used).
    !!! However:
    notice that `1 == True` and `0 == False` evaluates to `True` what is a reason behind
        coalesce0(None, False, 0, 1, 0, empty=(None, False))    # 1
    (what is WRONG!)
    but
        coalesce(None, False, 0, 1, 0, empty=(None, False))     # 0
    what is demanded behaviour.

    Thus this version may be safely used as long as we do not want to distinguish
    between 0 and False or 1 and True; see examples.

    coalesce0(1, None, None)     # 1
    coalesce0(None, None, None)  # None
    coalesce0(None, None, False) # False
    coalesce0(None, None, False, empty=(None, False))    # None
    coalesce0(None, None, False, empty=(False, None))    # False
    coalesce0(None, None, False, 1, empty=(None, False)) # 1
    coalesce0(None, False, 0, 1, 0, empty=(None, False)) # 1     # because `0 == False` is True
    coalesce(None, False, 0, 1, 0, empty=(None, False))  # 0       # because `0 is False` is False
    """
    for res in args:
        if res not in empty:
            return res
    return empty[0]


def coalesce(*args, empty: tuple[Any] = (None,)) -> Any | None:
    """
    As in PostgreSQL: returns first not None argument;
    if all arguments are None then returns None.

    However it is possible to provide own empties via `empty` keyword-argument:
    each element of `empty` tuple is considered as being "empty" value
    and is ommited when searching for first non-empty.
    If all arguments are empty then first element of `empty` (None by default) is returned.

    This is safe (thus default to use) version
    where we use `is` operator for checking if element is in `empty`.
    Notice that `1 == True` and `0 == False` evaluates to `True` what is a reason behind
        coalesce0(None, False, 0, 1, 0, empty=(None, False))     # 1, as `0 == False` is True
    (what is WRONG!)
    but
        coalesce(None, False, 0, 1, 0, empty=(None, False))      # 0, as `0 is False` is False
    what is demanded behaviour.

    However this version is slower (checking `is` is slower then `==` ?)
    then coalesce0()

    coalesce(1, None, None)     # 1
    coalesce(None, None, None)  # None
    coalesce(None, None, False) # False
    coalesce(None, None, False, empty=(None, False))    # None
    coalesce(None, None, False, empty=(False, None))    # False
    coalesce(None, None, False, 1, empty=(None, False)) # 1
    coalesce(None, False, 0, 1, 0, empty=(None, False)) # 0       # because `0 is False` is False
    coalesce0(None, False, 0, 1, 0, empty=(None, False))  # 1     # because `0 == False` is True
    """
    for res in args:
        #if not any(res is e for e in empty):
        #   return res
        broken = False
        for e in empty:
            if e is res:
                broken = True
                break
        if not broken:
            return res
    return empty[0]

"""
%timeit coalesce1(1, None, None)
483 ns ± 3.49 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit coalesce0(1, None, None)
127 ns ± 1.42 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit coalesce(1, None, None)
147 ns ± 0.993 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)  -- version with `for e in empty`
336 ns ± 2.53 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)    -- version with `if not any`

lst = [None]*100 + [1]
%timeit coalesce1(*lst)
5.52 µs ± 141 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce0(*lst)
2.46 µs ± 99.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce(*lst)
5.36 µs ± 49.5 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)  -- version with `for e in empty`
29.1 µs ± 581 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)    -- version with `if not any`

lst2 = lst + [1]*100
%timeit coalesce1(*lst2)
12.4 µs ± 71 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce0(*lst2)
3.03 µs ± 75.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce(*lst2)
6.46 µs ± 245 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each) -- version with `for e in empty`
29.7 µs ± 218 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)  -- version with `if not any`
"""

# %%
def filter_first(*args, predicate: callable = bool, default: Any = None) -> Any | None:
    """
    More general form of coalesce():
    it returns first argument which satisfies `predicate`,
    i.e. for which callable `predicate` returns True.
    By default `predicate = bool` i.e. we check if the argument evaluates to True (via inbuilt bool()).

    filter_first(1, None, None)             # 1
    filter_first(None, None, None)          # None
    filter_first(None, False, 0, 1, 0)      # 1
    coalesce(None, False, 0, 1, 0, empty=(None, False, 0))     # 1
    filter_first(None, False, 0, 1, 0, predicate = lambda x: x is not None)     # False
    coalesce(None, False, 0, 1, 0)          # False
    filter_first(None, False, 0, 1, 0, predicate = lambda x: x < 1)
        # ! TypeError: '<' not supported between instances of 'NoneType' and 'int'
    filter_first(None, False, 0, 1, 0, predicate = lambda x: x is not None and x < 1)   # False
    filter_first(False, True, False, 0, 1, 0)       # True
    filter_first(True, False, 0, 1, 0, predicate = lambda x: not x)   # False
    coalesce(True, False, 0, 1, 0, empty=(True,))   # False
    """
    for res in args:
        if predicate(res):
            return res
    return default


# %%
def dict_default(dic: dict, field: str, default: Any) -> Any:
    """
    Defult value from dict field if it exists but is empty.
    dict_default(dic, field, default) == coalesce(dic.get(field), default)
    but is faster then coalesce()
    dic = dict(a=1, b=None)
    dict_default(dic, 'a', 0)   # 1
    dict_default(dic, 'b', 0)   # 0
    dict_default(dic, 'c', 2)   # 0
    Notice that
    dict_default(dic, field, default) != dic.get(field) or default
    in a cases like:
    dic = dict(a=1, b=None, c=0)
    dict_default(dic, 'c', False)   # 0
    dic.get('c') or False           # False
    """
    value = dic.get(field, default)
    value = value if value is not None else default     # == coalesce(value, default)   != value or default
    return value


"""
dic = dict(a=1, b=None)
%timeit dict_default(dic, 'b', 0)
102 ns ± 1.71 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit dic.get('b') or 0
%timeit coalesce(dic.get('b'), 0)
230 ns ± 5.4 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit coalesce0(dic.get('b'), 0)
168 ns ± 1.02 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

%timeit dict_default(dic, 'c', 2)
98.2 ns ± 1.58 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit coalesce(dic.get('c'), 2)
232 ns ± 3.61 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit coalesce0(dic.get('c'), 2)
168 ns ± 0.69 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
"""


# %%
def partial_match(s: str, full: str, value: bool | int = True) -> str | None:
    """
    Almost synonym for `full.startswith(s)` but may also return `full` or None (not only True/False)
    partial_match('qq', 'qqryq', False)    # True
    partial_match('qq', 'kukuryq', 0)  # False
    partial_match('qq', 'qqryq')    # 'qqryq'
    partial_match('qq', 'kukuryq', 1)  # None
    """
    match = full.startswith(s)
    if value:
        res = full if match else None
    else:
        res = match
    return res


def first_match(s: str, fulls: list[str]) -> str | None:
    """ this is really useful: resutrn first of `fulls` which starts with `s`
    first_match('qq', ['aqq', 'ryq', 'qq-ryq', 'hop', 'qq-ryq-2'])  # 'qq-ryq'
    first_match('aqq', ['qq', 'ryq'])      # None
    first_match('aqq', [])                 # None
    """
    res = filter_first(*fulls, predicate=lambda z: z.startswith(s))
    return res


def all_matches(s: str, fulls: list[str]) -> list[str]:
    """
    all_matches('qq', ['aqq', 'ryq', 'qq-ryq', 'hop', 'qq-ryq-2'])  # ['qq-ryq', 'qq-ryq-2']
    all_matches('qq', ['aqq', 'ryq', 'qq-ryq', 'hop'])              # ['qq-ryq']
    all_matches('qq', ['aqq', 'ryq', 'hop'])                        # []
    all_matches('qq', [])                                           # []
    """
    res = [z for z in fulls if z.startswith(s)]
    return res


# %%
def timeit(fun):
    """decorator for timing"""
    @wraps(fun)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fun(*args, **kwargs)
        t1 = time.time()
        dt = t1 - t0
        print("Execution time: {:3.5f}".format(dt))
        return result
    return wrapper


# %%
def where_up(path: Union[str, PosixPath], pattern: str) -> PosixPath:
    """
    Finds first directory upper wrt. `path` (one of parents in terms of pathlib)
    containing file or dir with a name which conforms to `pattern`.
    E.g.  where_up(pth, '.git')  finds root repository directory in which `pth` resides (whatever deep).
    path: str / PosixPath;
    pattern: str; glob pattern, see: ...
    """
    path = Path(path).absolute()
    while not list(path.glob(pattern)) and path != Path("/"):
        path = path.parent
    return path


# %%
def replace_deep(obj, what, to):
    """
    Replacing values in `obj` given by `what` with value given by `to`;

    `obj` in general is json-like, i.e. may be dictionary or list (or simple type)
    elements of wihch are dictionaries or lists (or simple types) and so on.
    Procedure is digging `obj` for replacing items in `what` with value of `to`;

    If value of a `obj` is a dictionary it is searched deeper for `what`s (recursively);
    if it's a list its elements are searched for `what`s (also recursively).
    if value of a `obj` is simple type then it's checked against `what`s
    and replaced with `to` if check is positive.

    `what` is iterable of items to be replaced; they cannot be lists or dicts.
    if `what` is a string then it is treated as iterable so to replace 'str' with some value
    one must pass ['str'] to `what`.

    Examples
    --------
    dd = {'a': ['', 0, 1], 'b': '', 'c': {0: '', 1: [3, 4, 'none', {'p': '', 'q': 'qq'}]}, 'd': None}
    replace_deep(dd, what='', to=None)      # ! TypeError: ...
    replace_deep(dd, what=[''], to=None)
    replace_deep(dd, what=[None], to=0)
    replace_deep(dd, what=(0, None), to='')
    replace_deep(dd, what=('', 'none'), to=None)
    replace_deep('', what=('', 'none'), to='None')
    replace_deep(1, what=('', 'none'), to=None)
    """
    o = deepcopy(obj)
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = replace_deep(v, what, to)
    elif isinstance(o, list):
        o = [replace_deep(l, what, to) for l in o]
    else:
        o = to if o in what else o
    return o


# %%
def dict_depth(dic: dict):
    """
    `dict` is tree-like with all leaves being simple types (not collections),
    e.g. str, numeric or date.
    This procedure checks the depth of this tree
    BUT only at the 1st-1st-...-1st branch,
    what stems from assumption that the tree is uniform
    in the meaning that all branches have the same length.

    Example
    -------
    dic = {'a': {'c': {'d': [1,2]}}}
    dict_depth(dic)  # 3
    """
    res = 1
    v = dic[list(dic.keys())[0]]
    if isinstance(v, dict):
        res += dict_depth(v)
    return res


# %%
def adaptive_round(value: Union[float, Iterable[float], str], r: int = 4):
    """
    Rounding numbers to only r significant digits;
    if not number returns value unchanged;
    iterates over elements of Iterables (recursively) except strings which are ignored.
    Notice the difference:
    adaptive_round(123456, 3)     # 123456
    "{:.3g}".format(123456)         # '1.23e+05'
    adaptive_round(.123456, 3)     # .1235
    "{:.3g}".format(.123456)         # '.123'
    adaptive_round(.000123456, 3)     # .0001235
    "{:.3g}".format(.000123456)         # '.000123'
    adaptive_round(.000000123456, 3)     # 1.235e-07
    "{:.3g}".format(.000000123456)         # '1.23e-07'
    """
    if value is None:
        pass
    elif isinstance(value, str):
        pass
    elif isinstance(value, Iterable):
        value = [adaptive_round(v, r) for v in value]
    else:
        try:
            r = round(m.log10(abs(value))) - r
            r = max(0, -r)
            value = round(value, r)
        except ValueError:
            pass
    return value


# %%
def fill_from_str(
        s: str | None,
        *args: Any | None,
        split: str = "-",
        longest: bool = False,
) -> tuple[str | None, ...]:
    """

    Args
    ----
    s: str
        string which will be split according to `split` sign
    *args: str
        any number of objects af any type (or None if given object is None);
    split: str = "-"
        sign wrt which the string `s` will be split
    longest: bool = False
        if False then returns tuple of length `min(length(s.split(split)), length(args))`;
        if True then returns tuple of length `max(length(s.split(split)), length(args))`.

    Returns
    -------
    `args` (as tuple) where each element which was empty (somehow) is filled
    with the respective element of `s` split wrt `split`.

    Examples
    --------
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '', '')                     # ('a', 'b/c', 'd', '  ', 'q/q')
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '')                         # ('a', 'b/c', 'd', '  ')
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '', longest=True)           # ('a', 'b/c', 'd', '  ', 'q/q')
    # "too many" `args`
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '', '', '')                 # ('a', 'b/c', 'd', '  ', 'q/q')
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '', '', '', longest=True)   # ('a', 'b/c', 'd', '  ', 'q/q', '')
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '', '', None, longest=True)   # ('a', 'b/c', 'd', '  ', 'q/q', '')
    fill_from_str('a-b/c-d-  -q/q', '', '', '', '', '', split="/")          # ('a-b', 'c-d-  -q', 'q')
    fill_from_str('a-b/c-d-  -q/q', '00', '', '11', split="/")              # ('00', 'c-d-  -q', '11')
    fill_from_str('a-b/c-d-  -q/q', '00', None, 11, False, 0)               # ('00', 'b/c', 11, '  ', 'q/q')
    fill_from_str('a--d-  -q/q', '00', None, 11, False, 0)                  # ('00', '', 11, '  ', 'q/q')
    """
    if not all(args) and s:
        ss = s.split(split)
        if longest:
            args = tuple((a or s) or '' for a, s in zip_longest(args, ss))
        else:
            args = tuple(a or s for a, s in zip(args, ss))
    return args


# %%
def forward_fill(
        lst: list,
        empty: tuple[Any | None, ...] | Any | None = (None,)
) -> list:
    """
    Works like pd.DataFrame.ffill().
    Transforms `lst` like [None, 'a', None, 'b', None, 'c', None]
    into ['a', 'a', 'a', 'b', 'b, 'c', 'c'].
    Notice that empties preceding first non-empty are filled with this first non-empty,
    i.e. for leading empties it works like back-fill.

    lst: list
        list to be forward filled;
    empty: tuple = (None,)
        what values are considered to be "empty" i.e. to be filled with "non-empty" preceding value from `lst`;
        e.g. if `empty = ''` then only empty strings will be filled while `None`s will be left intact.

    Examples
    --------
    forward_fill(['', 'a', '', 'b', '', 'c', ''])           # ['', 'a', '', 'b', '', 'c', '']
    forward_fill(['', 'a', '', 'b', '', 'c', ''], '')       # ['a', 'a', 'a', 'b', 'b', 'c', 'c']
    forward_fill(['', 'a', '', 'b', '', 'c', None], '')     # ['a', 'a', 'a', 'b', 'b', 'c', None]
    forward_fill(['', 'a', '', 'b', '', 'c', None])                 # ['', 'a', '', 'b', '', 'c', 'c']
    forward_fill(['', 'a', '', 'b', '', 'c', None], (None, ''))     # ['a', 'a', 'a', 'b', 'b', 'c', 'c']
    #
    forward_fill(['', 'a', '', 'b', '', 'c'], '')           # ['a', 'a', 'a', 'b', 'b', 'c']
    forward_fill(['', '', '', '', '', 'c'], '')             # ['c', 'c', 'c', 'c', 'c', 'c']
    forward_fill(['', '', '', '', '', ''])                  # ['', '', '', '', '', '']
    forward_fill(['', '', '', '', '', ''], '')              # ['', '', '', '', '', '']
    forward_fill(['', '', '', '', '', ''], (None, ''))      # ['', '', '', '', '', '']
    #
    forward_fill([0, 1, 0, 0, None, 2, 0])                  # [0, 1, 0, 0, 0, 2, 0]
    forward_fill([0, 1, 0, 0, None, 2, 0], (0,))            # [1, 1, 1, 1, None, 2, 2]
    """

    if not isinstance(empty, tuple):
        empty = (empty,)

    # find first non-empty
    v = lst[0]
    k = 0
    while k < len(lst):
        if lst[k] not in empty:
            v = lst[k]
            break
        else:
            k += 1

    if v in empty:
        pass
    else:
        # fill
        k = 0
        while k < len(lst):
            if lst[k] not in empty:
                v = lst[k]
            else:
                lst[k] = v
            k += 1

    return lst


ffill = forward_fill


def backward_fill(
        lst: list,
        empty: tuple[Any | None, ...] | Any | None = (None,)
) -> list:
    """
    Works like pd.DataFrame.bfill().
    Transforms `lst` like [None, 'a', None, 'b', None, 'c', None]
    into ['a', 'a', 'b', 'b', 'c, 'c', 'c'].
    Notice that empties following last non-empty are filled with this last non-empty,
    i.e. for trailing empties it works like forward-fill.

    lst: list
        list to be backward filled;
    empty: tuple = (None,)
        what values are considered to be "empty" i.e. to be filled with "non-empty" following value from `lst`;
        e.g. if `empty = ""` then only empty strings will be filled while `None`s will be left intact.

    Examples
    --------
    backward_fill(['', 'a', '', 'b', '', 'c', ''])           # ['', 'a', '', 'b', '', 'c', '']
    backward_fill(['', 'a', '', 'b', '', 'c', ''], '')       # ['a', 'a', 'b', 'b', 'c', 'c', 'c']
    backward_fill(['', 'a', '', 'b', '', 'c', None], '')     # ['a', 'a', 'b', 'b', 'c', 'c', None]
    backward_fill(['', 'a', '', 'b', '', 'c', None])                 # ['', 'a', '', 'b', '', 'c', 'c']
    backward_fill(['', 'a', '', 'b', '', 'c', None], (None, ''))     # ['a', 'a', 'b', 'b', 'c', 'c', 'c']
    #
    backward_fill(['', 'a', '', 'b', '', 'c'], '')           # ['a', 'a', 'a', 'b', 'c', 'c']
    backward_fill(['', '', '', '', '', 'c'], '')             # ['c', 'c', 'c', 'c', 'c', 'c']
    backward_fill(['a', '', '', '', '', ''], '')             # ['a', 'a', 'a', 'a', 'a', 'a']
    backward_fill(['', '', '', '', '', ''])                  # ['', '', '', '', '', '']
    backward_fill(['', '', '', '', '', ''], '')              # ['', '', '', '', '', '']
    backward_fill(['', '', '', '', '', ''], (None, ''))      # ['', '', '', '', '', '']
    #
    backward_fill([0, 1, 0, 0, None, 2, 0])                  # [0, 1, 0, 0, 2, 2, 0]
    backward_fill([0, 1, 0, 0, None, 2, 0], 0)               # [1, 1, None, None, None, 2, 2]
    """
    return forward_fill(lst[::-1], empty)[::-1]


bfill = backward_fill


# %%
def lengthen(x: Any, n: int, as_list: bool = True) -> Union[list, tuple]:
    """
    Lengthening ll to the len(ll) == n;
    if ll is not iterable then it is turned into 1 element list (if `as_list` is True)
    or tuple (in other case) and then 'multiplied' n times.
    if len(ll) > n then ll is in fact shortened to have length == n.
    `as_list` is irrelevant in case of x is iterable.

    Examples
    --------
    lengthen([1, 3, 2, 5], 7, True)     # [1, 3, 2, 5, 5, 5, 5]
    lengthen([1, 3, 2, 5], 3, False)    # [1, 3, 2]
    lengthen([1, 3, 2, 5], 0)           # []
    lengthen(1, 3)                      # [1, 1, 1]
    lengthen(1, 3, False)               # (1, 1, 1)
    lengthen(1, 1)                      # [1]
    lengthen('ab', 3)                   # ['ab', 'ab', 'ab']
    """
    if isinstance(x, (list, tuple)):
        is_list = isinstance(x, list)
        l = len(x)
        if l < n:
            if is_list:
                x = x + [x[-1]] * (n - l)
            else:
                x = x + (x[-1],) * (n - l)
        else:
            x = x[:n]
    elif isinstance(x, (int, float, str)):
        if as_list:
            x = [x] * n
        else:
            x = (x,) * n
    else:
        raise Exception("`x` must be in (int, float, str) or iterable of these.")

    return x


# %%
def lengthen0(x: Any, n: int, as_list: bool = True) -> Union[list, tuple]:
    """
    better (?) version of lengthen0()

    Examples
    --------
    lengthen0([1, 3, 2, 5], 7, True)     # [1, 3, 2, 5, 5, 5, 5]
    lengthen0([1, 3, 2, 5], 3, False)    # [1, 3, 2]
    lengthen0([1, 3, 2, 5], 0)           # 0
    lengthen0(1, 3)                      # [1, 1, 1]
    lengthen0(1, 3, False)               # (1, 1, 1)
    lengthen0(1, 1)                      # [1]
    lengthen0('ab', 3)                   # ['ab', 'ab', 'ab']
    """
    if isinstance(x, (list, tuple)):
        is_list = isinstance(x, list)
        x = it.islice(it.chain(x, it.repeat(x[-1], max(0, n - len(x)))), n)

        if is_list:
            x = list(x)
        else:
            x = tuple(x)

    elif isinstance(x, (int, float, str)):
        if as_list:
            x = [x] * n
        else:
            x = (x,) * n
    else:
        raise Exception("`x` must be in (int, float, str) or iterable of these.")

    return x


"""
%timeit lengthen([1, 3, 2, 5], 7, True)
350 ns ± 17.6 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit lengthen0([1, 3, 2, 5], 7, True)
787 ns ± 36.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
# it's clearly seen that itertools approach is slower (over 2 times).
"""


# %%
def flatten(lst: list) -> list:
    """
    Examples
    --------
    flatten([1, 2, [[3, 4, [5]], 6, 7], 8]) == [1, 2, 3, 4, 5, 6, 7, 8]
    flatten([1, 2, [["3, 4", [5]], 6, 7], "8"]) == [1, 2, "3, 4", 5, 6, 7, "8"]
    flatten([1, 2, [[(3, 4), [5]], {6, 7}], "8"]) == [1, 2, (3, 4), 5, {6, 7}, "8"]
    """
    result = []
    for l in lst:
        if isinstance(l, list):
            result.extend(flatten(l))
        else:
            result.append(l)
    return result


# %%
def paste(left, right,
          by_left: bool = True, sep: str = "", flat: bool = True):
    """
    Works similar to R's paste().
    by_left: bool

    Examples
    --------
    paste(['a', 'b'], [1, 2])               # ['a1', 'a2', 'b1', 'b2']
    paste(['a', 'b'], [1, 2], False)        # ['a1', 'b1', 'a2', 'b2']
    paste(['a', 'b'], [1, 2], flat=False)           # [['a1', 'a2'], ['b1', 'b2']]
    paste(['a', 'b'], [1, 2], False, flat=False)    # [['a1', 'b1'], ['a2', 'b2']]
    paste(['a', 'b'], [1, 2], sep="_")      # ['a_1', 'a_2', 'b_1', 'b_2']
    """
    result = []

    if not isinstance(left, list) and isinstance(left, (range, tuple, set, dict)):
        left = list(left)
    if not isinstance(right, list) and isinstance(right, (range, tuple, set, dict)):
        right = list(right)

    def l_sep_r(l, r):
        return f"{l}{sep}{r}"

    if isinstance(left, list) and isinstance(right, list):
        if by_left:
            result = [[l_sep_r(l, r) for r in right] for l in left]
        else:
            result = [[l_sep_r(l, r) for l in left] for r in right]

        if flat:
            result = flatten(result)

    elif isinstance(left, list):
        result = [l_sep_r(l, right) for l in left]

    elif isinstance(right, list):
        result = [l_sep_r(left, r) for r in right]

    else:
        result = l_sep_r(left, right)

    return result


# %%
def union(*args):
    return reduce(lambda a, b: set(a).union(b), args, set())


def dict_set_union(dic1: Dict[Any, Set], dic2: Dict[Any, Set]) -> Dict[Any, Set]:
    """
    dict1, dict2: dictionaries of sets !!!
    Returns dictionary with all the keys from both dicts
    and union of sets for each key.

    Examples
    --------
    dic1 = {"a":{1, 2}, "b":{0, "0"}, "c":set()}
    dic2 = {"a":{3, 5, 7}, "c":{0, 2}, "d":{"qq"}}

    dict_set_union(dic1, dic2)

    union(*dic2.values())
    union(*dic2.values(), *dic1.values())
    """
    keys = set(dic1.keys()).union(dic2.keys())

    dic_new = dict()
    for k in keys:
        try:
            set1 = set(dic1.get(k, set()))
            set2 = set(dic2.get(k, set()))
            # ! Remark: Python 3.9 (at least):  {} is empty dict  not  empty set  (!)
            # but  set({})  is empty set (like just  set() )
            dic_new[k] = set1.union(set2)
        except Exception as e:
            print(e)

    return dic_new


# %%
def dict_list_sum(dic1: Dict[Any, list], dic2: Dict[Any, list]) -> Dict[Any, list]:
    """
    dict1, dict2: dictionaries of lists !!!
    Returns dictionary with all the keys from both dicts
    and sum of lists for each key.

    Examples
    --------
    dic1 = {"a":[1, 2], "b":[0, "0"], "c":[], "d": "ryq"}
    dic2 = {"a":[3, 5, 7], "c":[0, 2], "d":["qq"]}

    dict_list_sum(dic1, dic2)
    """
    keys = set(dic1.keys()).union(dic2.keys())

    dic_new = dict()
    for k in keys:
        try:
            l1 = dic1.get(k, [])
            l2 = dic2.get(k, [])
            l1 = l1 if isinstance(l1, list) else [l1]
            l2 = l2 if isinstance(l2, list) else [l2]
            dic_new[k] = l1 + l2
        except Exception as e:
            print(e)

    return dic_new


def dict_list_sum_reduce(ldl: List[Dict[Any, List]]) -> Dict[Any, List]:
    """
    ldl: list of dictionaries of lists
        all dicts should have the same keys (although it's not necessary).

    Result is a dictionary with union of all keys from each of dictionaries
    where for each key the value is a list being list-sum of all lists under this key
    across all the dictionaries in `ldl`.
    """
    res = reduce(dict_list_sum, ldl, dict())
    return res

# %%
