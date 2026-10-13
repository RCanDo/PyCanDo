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
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
# from __future__ import annotations
from pathlib import Path, PosixPath
import time
from copy import deepcopy

from functools import reduce, wraps
from itertools import zip_longest, islice, repeat, chain, product, cycle
from collections import defaultdict

import math as m

from collections.abc import Callable, Iterable, Sequence  # , Mapping
from typing import Union, Any, TypeVar  #, Optional, NewType,

T = TypeVar("T")


# %%
def iterable_not_str(x: Any) -> bool:
    """ also in  mytypes.py """
    return isinstance(x, Iterable) and not isinstance(x, str)


def all_not_none(*args: Any) -> bool:
	"""
	Checks if all of the arguments are not None.
	"""
	return all(t is not None for t in args)


# %%
def coalesce1(*args: Any, empty: tuple[Any] = (None,)) -> Any | None:
    """
    Depricated as coalesce() via loop is much faster;
    left for reference.

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


def coalesce0(*args: Any, empty: tuple[Any] = (None,)) -> Any | None:
    """
    As in PostgreSQL: returns first not None argument;
    if all arguments are None then returns None.

    However, it is possible to provide own empties via `empty` keyword-argument:
    each element of `empty` tuple is considered as being "empty" value
    and is ommited when searching for first non-empty.
    If all arguments are empty then first element of `empty` (None by default) is returned.

    !!! This version is faster then `coalesce()` but
    may be _safely_ used _only_ if we do not include in `empty` 0, 1, `False` and `True`,
    or, more precisely: do not want to distinguish between 0 and `False` or 1 and `True`.

    Namely:
    `==` operator is used here for checking if any of the arguments is in `empty`,
    i.e. we check `arg in empty` (for which `==` is used).
    !!! However:
    notice that `1 == True` and `0 == False` evaluates to `True` what is a reason behind
        coalesce0(None, False, 0, 1, 0, empty=(None, False))    # 1
    (what is WRONG!)
    but
        coalesce(None, False, 0, 1, 0, empty=(None, False))     # 0
    what is demanded behaviour.

    coalesce0(None, None, 1)            # 1
    coalesce0(None, None, 0, 1)         # 0
    coalesce0(1, None, None)            # 1
    coalesce0(None, None, None)         # None
    coalesce0(None, None, False)        # False
    coalesce0(None, None, False, empty=(None, False))   # None  – all are in `empty` thus returns `empty[0]`
    coalesce0(None, None, False, empty=(False, None))   # False –  "
    coalesce0(None, None, False, 1)                         # False
    coalesce0(None, None, False, 1, empty=(None, False))    # 1
    coalesce0(None, False, 0, 1, 0, empty=(None, False))    # 1     # !!!  because `0 == False` is True
    coalesce(None, False, 0, 1, 0, empty=(None, False))     # 0     # because `0 is False` is False
    """
    for arg in args:
        if arg not in empty:
            return arg
    return empty[0]


def coalesce(*args: Any, empty: tuple[Any] = (None,)) -> Any | None:
    """
    As in PostgreSQL: returns first not None argument;
    if all arguments are None then returns None.

    However, it is possible to provide own empties via `empty` keyword-argument:
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

    However, this version is slower (checking `is` is slower then `==` ?) than `coalesce0()`.

    Examples
    --------
    coalesce(None, None, 1)            # 1
    coalesce(None, None, 0, 1)         # 0
    coalesce(1, None, None)            # 1
    coalesce(None, None, None)         # None
    coalesce(None, None, False)        # False
    coalesce(None, None, False, empty=(None, False))    # None  – all are in `empty` thus returns `empty[0]`
    coalesce(None, None, False, empty=(False, None))    # False –  "
    coalesce(None, None, False, 1)                          # False
    coalesce(None, None, False, 1, empty=(None, False))     # 1
    coalesce(None, False, 0, 1, 0, empty=(None, False))     # 0     # because `0 is False` is False
    coalesce0(None, False, 0, 1, 0, empty=(None, False))    # 1     # !!! because `0 == False` is True
    """
    for arg in args:
        # if not any(arg is e for e in empty):   # alternative version – much slower
        #    return arg
        is_empty = False
        for e in empty:
            if e is arg:
                is_empty = True
                break
        if not is_empty:
            return arg
    return empty[0]

"""
%timeit coalesce1(1, None, None)    # 770 ns ± 26.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit coalesce0(1, None, None)    # 194 ns ± 1.07 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit coalesce(1, None, None)     # 226 ns ± 6.78 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
                                    # – version with `for e in empty`
                                    # 559 ns ± 15.5 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
                                    # – version with `if not any`
%timeit next(filter(lambda x: x is not None, [1, None, None]))
                                    # 329 ns ± 5.62 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

lst = [None]*100 + [1]
%timeit coalesce1(*lst)             # 9.05 μs ± 243 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce0(*lst)             # 3.13 μs ± 103 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce(*lst)              # 7.58 μs ± 245 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
                                    # – version with `for e in empty`
                                    # 71.8 μs ± 1.09 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
                                    # – version with `if not any`
%timeit next(filter(lambda x: x is not None, lst))
                                    # 6.88 μs ± 187 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

lst2 = lst + [1]*100
%timeit coalesce1(*lst2)            # 21.1 μs ± 323 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce0(*lst2)            # 3.87 μs ± 51.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit coalesce(*lst2)             # 8.62 μs ± 165 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
                                    # – version with `for e in empty`
                                    # 71.1 μs ± 2.25 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
                                    # – version with `if not any`
%timeit next(filter(lambda x: x is not None, lst2))
                                    # 6.92 μs ± 156 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
"""

# %%
def filter_first(*args: Any, predicate: Callable[[Any], bool] = bool, default: Any = None) -> Any | None:
    """
    More general form of coalesce():
    it returns first argument which satisfies `predicate`,
    i.e. for which callable `predicate` returns True.
    By default `predicate = bool` i.e. we check if the argument evaluates to True (via inbuilt bool()).
    This is the same as
        next(filter(predicate, [*args]))
    or
        list(filter(predicate, [*args]))[0]
    but faster for long `args`.

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

"""
%timeit filter_first(None, False, 0, 1, 0)
                        # 290 ns ± 4.36 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit next(filter(bool, [None, False, 0, 1, 0]))
                        # 212 ns ± 3.65 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit list(filter(bool, [None, False, 0, 1, 0]))[0]
                        # 343 ns ± 2.98 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

lst = [None]*100 + [1]
%timeit filter_first(lst)           # 195 ns ± 0.769 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit next(filter(bool, lst))     # 834 ns ± 27.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit list(filter(bool, lst))[0]  # 980 ns ± 29.3 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

lst2 = lst + [1]*100
%timeit filter_first(lst2)           # 195 ns ± 0.113 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit next(filter(bool, lst2))     # 864 ns ± 19 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit list(filter(bool, lst2))[0]  # 3.24 _μs_ ± 20.2 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

%timeit filter_first(None, False, 0, 1, 0, predicate = lambda x: x is not None and x < 1)
                                    # 398 ns ± 10.8 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit next(filter(lambda x: x is not None and x < 1, [None, False, 0, 1, 0]))
                                    # 432 ns ± 11.7 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit list(filter(lambda x: x is not None and x < 1, [None, False, 0, 1, 0]))[0]
                                    # 819 ns ± 26 ns ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit filter(lambda x: x is not None and x < 1, [None, False, 0, 1, 0])
                                    # 245 ns ± 7.72 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
                                    # but this is just generator object – not a final value
"""

# %%
def dict_default(dic: dict[str, Any], field: str, default: Any, empty: tuple[Any] = (None,)) -> Any:
    """
    Default value from dict field if it exists but is one of `empty` (which by default is `(None,)`).
    dict_default(dic, field, default, empty) == coalesce(dic.get(field), default, empty)
    but is faster then coalesce().

    dic = dict(a=1, b=None)
    dict_default(dic, 'a', 0) 	 	 	# 1
    dict_default(dic, 'a', 0, (1,)) 	 	# 0
    dict_default(dic, 'b', 0)    	 	# 0
    dict_default(dic, 'b', 0, ("",)) 	# None
    dict_default(dic, 'c', 2) 	 	 	# 2
    dict_default(dic, 'c', 2, ("",))  	# 2
    # Notice that
	    dict_default(dic, field, default) != dic.get(field) or default
    # in cases like:
    dic = dict(a=1, b=None, c=0)
    dict_default(dic, 'c', False)   # 0
    dic.get('c') or False           # False
    """
    value = dic.get(field, default)
    value = value if value not in empty else default     # == coalesce(value, default)   != value or default
    return value


"""
dic = dict(a=1, b=None)
%timeit dict_default(dic, 'b', 0)   # 118 ns ± 1.13 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit dic.get('b') or 0           # 55.8 ns ± 1.63 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
                                    # faster but not always the same result, see axamples
%timeit coalesce0(dic.get('b'), 0)  # 249 ns ± 0.29 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit coalesce(dic.get('b'), 0)   # 323 ns ± 5.84 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

%timeit dict_default(dic, 'c', 2)   # 138 ns ± 6.29 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
%timeit coalesce(dic.get('c'), 2)   # 337 ns ± 6.52 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit coalesce0(dic.get('c'), 2)  # 253 ns ± 4.65 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
"""

# %%
def partial_match(s: str, full: str, value: bool = True) -> str | None:
    """
    Almost synonym for `full.startswith(s)` but may also return `full` or `None` (not only `True/False`).
    partial_match('qq', 'qqryq')           # 'qqryq'
    partial_match('qq', 'qqryq', True)     # 'qqryq'
    partial_match('qq', 'qqryq', False)    # True
    partial_match('qq', 'kukuryq')         # None
    partial_match('qq', 'kukuryq', 0)      # False
    """
    match = full.startswith(s)
    if value:
        res = full if match else None
    else:
        res = match
    return res


def first_match(s: str, fulls: Iterable[str]) -> str | None:
    """
    Returns first of `fulls` which starts with `s`.
    The same as
        next(filter(lambda z: re.search(f"^{s}", z), fulls))
    but faster.

    first_match('qq', ['aqq', 'ryq', 'qq-ryq', 'hop', 'qq-ryq-2'])  # 'qq-ryq'
    first_match('aqq', ['qq', 'ryq'])      # None
    first_match('aqq', [])                 # None
    """
    res = filter_first(*fulls, predicate=lambda z: z.startswith(s))
    return res


def all_matches(s: str, fulls: Iterable[str]) -> tuple[str]:
    """
    Returns tuple of all `fulls` which starts with `s`.
    The same as
        list(filter(lambda z: re.search(f"^{s}", z), fulls))
    but faster.

    all_matches('qq', ['aqq', 'ryq', 'qq-ryq', 'hop', 'qq-ryq-2'])  # ('qq-ryq', 'qq-ryq-2')
    all_matches('qq', ['aqq', 'ryq', 'qq-ryq', 'hop'])              # ('qq-ryq')
    all_matches('qq', ['aqq', 'ryq', 'hop'])                        # ()
    all_matches('qq', [])                                           # ()
    """
    res = tuple(z for z in fulls if z.startswith(s))
    return res

"""
import re
s = 'qq'
fulls = ['aqq', 'ryq', 'qq-ryq', 'hop', 'qq-ryq-2']
%timeit first_match(s, fulls)
                    # 979 ns ± 31 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit next(filter(lambda z: re.search(f"^{s}", z), fulls))
                    # 2.63 μs ± 54 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

%timeit all_matches(s, fulls)
                    # 731 ns ± 33.2 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit list(filter(lambda z: re.search(f"^{s}", z), fulls))
                    # 4.29 μs ± 137 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
"""

# %%
def timeit[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    """decorator for timing"""
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        dt = t1 - t0
        print("Execution time: {:3.5f}".format(dt))
        return result
    return wrapper


# %%
def where_up(path: str | PosixPath, pattern: str) -> PosixPath:
    """
    Finds first directory upper wrt. `path` (one of parents in terms of pathlib)
    containing file or dir with a name which conforms to `pattern`.
    E.g.  where_up(pth, '.git')  finds root repository directory in which `pth` resides (whatever deep).
    path: str / PosixPath;
    pattern: str; glob pattern, see: ...

    where_up('.', '.git')
    """
    path = Path(path).absolute()
    while not list(path.glob(pattern)) and path != Path("/"):
        path = path.parent
    return path


# %%
def replace_deep(obj: Any, what: Any, to: Any) -> Any:
    """
    Replacing values in `obj` given by `what` with value given by `to`;

    `obj` in general is json-like, i.e. may be dictionary or list (or simple type)
    elements of wihch are dictionaries or lists (or simple types) and so on.
    Procedure is digging `obj` for replacing items in `what` with value of `to`;

    If value of an `obj` is a dictionary it is searched deeper for `what`s (recursively);
    if it's a list its elements are searched for `what`s (also recursively).
    if value of an `obj` is simple type then it's checked against `what`s
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
def dict_depth(dic: dict) -> int:
    """
    `dict` is tree-like where all leaves are not dictionary.
    This procedure checks the depth of this tree
    BUT only at the 1st-1st-...-1st branch,
    what stems from assumption that the tree is uniform
    in the meaning that all branches have the same length.

    Example
    -------
    dict_depth(7)           # ! AttributeError: 'int' object has no attribute 'keys'
    dict_depth({'a': 7})    # 1
    dict_depth({'a': {'c': {'d': {1, 2}}}})  # 3
    dict_depth({'a': {'c': {'d': [1, 2]}}})  # 3
    dict_depth({'a': {'c': {'d': [1, [2, 0]]}}})  # 3
    dict_depth({'a': {'c': {'d': {1: [2, 0]}}}})  # 4
    """
    if not isinstance(dic, dict):
        return 0
    res = 1
    v = dic[tuple(dic.keys())[0]]
    if isinstance(v, dict):
        res += dict_depth(v)
    return res


# %%
U = TypeVar("U", float, str, None)

def adaptive_round(value: U | Iterable[U] | None, r: int = 4) -> U | list[U]:
    """
    Rounding numbers to only r or r+1 significant digits (depends on exact `value`);
    if not number returns value unchanged;
    iterates over elements of Iterables (recursively) except strings which are ignored.

    # Notice the difference:
    adaptive_round(123456, 3)     # 123456
    "{:.3g}".format(123456)       # '1.23e+05'
    adaptive_round(.123456, 3)      # .1235
    "{:.3g}".format(.123456)        # '.123'
    adaptive_round(.000123456, 3)     # .0001235
    "{:.3g}".format(.000123456)       # '.000123'
    adaptive_round(.000000123456, 3)    # 1.235e-07
    "{:.3g}".format(.000000123456)      # '1.23e-07'

    adaptive_round('0.123456', 3)   # '0.123456'
    adaptive_round([.123456, '0.123456', .00054321, None], 3)   # [0.1235, '0.123456', .00054321, None]
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
    s: str | None,
        string which will be split according to `split` sign
    *args: Any | None,
        any number of objects af any type (or None if given object is None);
    split: str = "-",
        sign wrt which the string `s` will be split
    longest: bool = False,
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
def lengthen0(x: Any, n: int, as_list: bool = True) -> Union[list, tuple]:
    """
    rather slow version of lengthen()
    as an example of applying itertools;

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
        x = islice(chain(x, repeat(x[-1], max(0, n - len(x)))), n)

        if is_list:
            x = list(x)
        else:
            x = tuple(x)

    elif isinstance(x, (int, float, str)):
        if as_list:
            x = [x] * n
        else:
            x = (x,) * n

    elif isinstance(x, (Iterable, Sequence)):
        x = lengthen(list(x), n, as_list)

    else:
        raise Exception("`x` must be in (int, float, str) or iterable of these.")

    return x


# %%
def lengthen(
        x: Union[T, Sequence[T], Iterable[T]],
        n: int,
        as_list: bool = None,
) -> Union[list[T], tuple[T, ...]]:
    """
    Lengthening x to the len(x) == n;
    if `x` is not list or tuple then
        if it is other Iterable or Sequence then it's turned into tuple;
        otherwise it is turned into 1 element tuple and then 'multiplied' n times.
    if len(ll) > n then x is in fact shortened to have length == n.
    If `as_list` is `None` then the result is returned as tuple and as a list only if `x` was a list.
    If `as_list` is `bool` then result is always turned to list if `True` and always to tuple if `False`.

    Examples
    --------
    # pass list
    lengthen([1, 3, 2, 5], 7)           # [1, 3, 2, 5, 5, 5, 5]
    lengthen([1, 3, 2, 5], 7, True)     # [1, 3, 2, 5, 5, 5, 5]
    lengthen([1, 3, 2, 5], 7, False)    # (1, 3, 2, 5, 5, 5, 5)    # the only case effect of `as_list=False` is visible

    # pass tuple
    lengthen((1, 3, 2, 5), 7)           # (1, 3, 2, 5, 5, 5, 5)
    lengthen((1, 3, 2, 5), 7, True)     # [1, 3, 2, 5, 5, 5, 5]
    lengthen((1, 3, 2, 5), 7, False)    # (1, 3, 2, 5, 5, 5, 5)

    # shortening
    lengthen([1, 3, 2, 5], 3)           # [1, 3, 2]
    lengthen([1, 3, 2, 5], 0)           # []

    # pass other Iterable / Sequence
    lengthen(range(4), 7)               # (0, 1, 2, 3, 3, 3, 3)
    lengthen(range(4), 7, as_list=True) # [0, 1, 2, 3, 3, 3, 3]

    # primitive
    lengthen(1, 3)                      # (1, 1, 1)
    lengthen(1, 3, True)                # [1, 1, 1]
    lengthen(1, 1)                      # (1,)
    lengthen('ab', 3)                   # ('ab', 'ab', 'ab')
    lengthen(['ab'], 3)                 # ['ab', 'ab', 'ab']
    """
    if isinstance(x, (list, tuple)):
        res: Union[list[T], tuple[T, ...]]
        l = len(x)
        if l < n:
            if isinstance(x, list):
                res = x + [x[-1]] * (n - l)
            else:
                res = x + (x[-1],) * (n - l)
        else:
            res = x[:n]
    elif isinstance(x, (Iterable, Sequence)) and not isinstance(x, str):
        res = lengthen(tuple(x), n, as_list)
    else:
        res = (x,) * n

    if as_list is not None:
        res = list(res) if as_list else tuple(res)

    return res


"""
%timeit lengthen([1, 3, 2, 5], 7, True)
568 ns ± 14.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit lengthen0([1, 3, 2, 5], 7, True)
1.06 μs ± 22.5 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
# it's clearly seen that itertools approach is slower (over 2 times).
"""


# %%
def align(*seqs: Any, mode: str = "lengthen", as_list: bool = True) -> list[list] | tuple[tuple, ...]:
    """
    Aligns all the sequences of `seqs` according to the `mode`:
    if `mode` is
    "lengthen" uses `align_lengthen(*seqs)`
    "prod"     uses `align_prod(*seqs)`
    "cycle"    uses `align_cycle(*seqs)`
    "lcm"      uses `align_cycle(*seqs, lcm=True)`.
    See help on these functions for details.
    If `as_list` then returns list of lists, otherwise (default) tuple of tuples
    (no matter what kind of sequences were passed).
    """
    if mode in "lengthen":
        res = align_lengthen(*seqs, as_list=as_list)
    elif mode in "prod":
        res = align_prod(*seqs, as_list=as_list)
    elif mode in "cycle":
        res = align_cycle(*seqs, as_list=as_list)
    elif mode in "lcm":
        res = align_cycle(*seqs, lcm=True, as_list=as_list)
    else:
        raise ValueError("`mode` must be one of 'lengthen', 'prod', 'cycle', 'lcm'.")
    return res


def align_lengthen(*seqs: Any, as_list: bool = False) -> list[list] | tuple[tuple, ...]:
    """
    Aligns all the sequences of `seqs` to the longest one
    by lengthening them (if shorter then the longest one).
    If `as_list` then returns list of lists, otherwise (default) tuple of tuples
    (no matter what kind of sequences were passed).

    seqs = [['a', 'b', 'c'], range(7), ['p', 'q']]
    align_lengthen(*seqs, as_list=False)
    # (('a', 'b', 'c', 'c', 'c', 'c', 'c'),
    #  ( 0,   1,   2,   3,   4,   5,   6),
    #  ('p', 'q', 'q', 'q', 'q', 'q', 'q'))

    align_lengthen(*seqs, as_list=True)
    # [['a', 'b', 'c', 'c', 'c', 'c', 'c'],
    #  [ 0,   1,   2,   3,   4,   5,   6],
    #  ['p', 'q', 'q', 'q', 'q', 'q', 'q']]
    """
    ml = max(len(s) for s in seqs)
    if not as_list:
        return tuple(lengthen(s, ml, as_list) for s in seqs)
    else:
        return [lengthen(s, ml, as_list) for s in seqs]


def align_prod(*seqs: Any, as_list: bool = False) -> list[list] | tuple[tuple, ...]:
    """
    Aligns sequences of `seqs` by making their cartesian product
    i.e. of all the values across sequences.
    If `as_list` then returns list of lists, otherwise (default) tuple of tuples
    (no matter what kind of sequences were passed).

    seqs = [range(3), range(2)]
    align_prod(*seqs)
    # ((0, 0, 1, 1, 2, 2),
    #  (0, 1, 0, 1, 0, 1))
    align_prod(*seqs, as_list=True)
    # [[0, 0, 1, 1, 2, 2],
    #  [0, 1, 0, 1, 0, 1]]

    seqs = [['a', 'b', 'c'], range(3), ['p', 'q']]
    align_prod(*seqs)
    # (('a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c'),
    #  ( 0,   0,   1,   1,   2,   2,   0,   0,   1,   1,   2,   2,   0,   0,   1,   1,   2,   2),
    #  ('p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q'))
    """
    if not as_list:
        return tuple(zip(*product(*seqs)))
    else:
        return [list(l) for l in zip(*product(*seqs))]


def align_cycle(*seqs: Any, lcm: bool = False, as_list: bool = False) -> list[list] | tuple[tuple, ...]:
    """
    Aligns all the sequences of `seqs` to the longest one
    by recycling the whole sequences (if shorter then the longest one).

    If `lcm = True` then
    aligns all the sequences of `seqs` to the least common multiplicity (lcm) of their lengths
    so that each sequence is always fully cycled.

    If `as_list` then returns list of lists, otherwise (default) tuple of tuples
    (no matter what kind of sequences were passed).

    seqs = [['a', 'b', 'c'], range(7), ['p', 'q']]
    align_cycle(*seqs)
    # (('a', 'b', 'c', 'a', 'b', 'c', 'a'),
    #  ( 0,   1,   2,   3,   4,   5,   6),
    #  ('p', 'q', 'p', 'q', 'p', 'q', 'p'))
    align_cycle(*seqs, as_list=True)
    # [['a', 'b', 'c', 'a'],
    #  [ 0,   1,   2,   3],
    #  ['p', 'q', 'p', 'q']]

    seqs = [['a', 'b', 'c'], range(4), ['p', 'q']]
    align_cycle(*seqs, lcm=True)
    (('a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'),
     ( 0,   1,   2,   3,   0,   1,   2,   3,   0,   1,   2,   3),
     ('p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q', 'p', 'q'))
    """
    lengths = [len(s) for s in seqs]
    ml = m.lcm(*lengths) if lcm else max(lengths)
    values = list(zip(*(cycle(s) for s in seqs), range(ml)))
    if not as_list:
        return tuple(zip(*values))[:-1]
    else:
        return [list(l) for l in zip(*values)][:-1]


# %%
def flatten(lst: list) -> list:
    """
    Flattens list `lst`, i.e. every nested list is unpacked, recursively,
    so that there is no list as element of the result (which is always a list).

    Examples
    --------
    flatten([1, 2, [[3, 4, [5]], 6, 7], 8])  # [1, 2, 3, 4, 5, 6, 7, 8]
    flatten([1, 2, [["3, 4", [5]], 6, 7], "8"])  # [1, 2, "3, 4", 5, 6, 7, "8"]
    flatten([1, 2, [[(3, 4), [5]], {6, 7}], "8"])  # [1, 2, (3, 4), 5, {6, 7}, "8"]
    """
    result = []
    for l in lst:
        if isinstance(l, list):
            result.extend(flatten(l))
        else:
            result.append(l)
    return result


# %%
def paste(
        left: Iterable | Sequence,
        right: Iterable | Sequence,
        by_left: bool = True,
        sep: str = "", flat: bool = True
) -> list[str | list[str]]:
    """
    Works similar to R's paste().
    by_left: bool

    Examples
    --------
    paste("ab", "xy")                       # 'abxy'
    paste("ab", "xy", sep="-")              # 'ab-xy'
    paste(list("ab"), "xy", sep="-")        # ['a-xy', 'b-xy']
    paste(list("ab"), list("xy"), sep="-")        # ['a-xy', 'b-xy']
    paste(["ab"], "xy", sep="-")            # ['ab-xy']
    paste(["ab"], ["xy"], sep="-")          # ['ab-xy']

    paste(['a', 'b'], [1, 2])               # ['a1', 'a2', 'b1', 'b2']
    paste(['a', 'b'], [1, 2], False)        # ['a1', 'b1', 'a2', 'b2']
    paste(['a', 'b'], [1, 2], flat=False)           # [['a1', 'a2'], ['b1', 'b2']]
    paste(['a', 'b'], [1, 2], False, flat=False)    # [['a1', 'b1'], ['a2', 'b2']]
    paste(['a', 'b'], [1, 2], sep="_")      # ['a_1', 'a_2', 'b_1', 'b_2']

    paste(list('ABC'), range(5), flat=False, sep="-")
    # [['A-0', 'A-1', 'A-2', 'A-3', 'A-4'],
    #  ['B-0', 'B-1', 'B-2', 'B-3', 'B-4'],
    #  ['C-0', 'C-1', 'C-2', 'C-3', 'C-4']]
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
def union(*args: Any) -> set:
    """
    Shorthand for
        set(arg).union(arg2, ..., argN)
    or
        set(arg) | set(arg2) | ... | set(argN);
    little slower then the former (it's just wrapper around it) but convenient.

    union([1,2,3], (2,4,6,8), range(0, 13, 3))  # {0, 1, 2, 3, 4, 6, 8, 9, 12}
    """
    return set(args[0]).union(*args[1:])

"""
%timeit union([1,2,3], (2,4,6,8), range(0, 13, 3))            # 579 ns ± 6.44 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit set([1,2,3]).union((2,4,6,8), range(0, 13, 3))        # 440 ns ± 7.13 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
%timeit set([1,2,3]) | set((2,4,6,8)) | set(range(0, 13, 3))  # 738 ns ± 5.72 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
"""


def dict_set_union(*sdics: dict[Any, set[Any]]) -> dict[Any, set[Any]]:
    """
    *sdics: dictionaries of sets;
        it's assumed that for each key the value is a set – otherwise TypeError is raised;
        use `dict_set_union_force()` if not all values are sets.

    Result is a dictionary with union of all keys from each of dictionaries
    where for each key the value is a set being set-union of all sets under this key
    across all the dictionaries in `sdics`.

    dic1 = {"a": {1, 2}, "b": {0, "0"}, "c": set(), "d": set("abc"), "e": {1}}
    dic2 = {"a": {3, 5, 7}, "b": {None}, "c": {0, 2}, "d": {"qq"}}
    dic3 = {"a": {0, 2, 5, 9}, "b": {0}, "e": {2}}

    dict_set_union(dic1, dic2, dic3)
    # {'a': {0, 1, 2, 3, 5, 7, 9},
    #  'b': {0, '0', None},
    #  'c': {0, 2},
    #  'd': {'a', 'b', 'c', 'qq'},
    #  'e': {1, 2}}

    dict_set_union(dic1)    # dic1
    dict_set_union()        # {}
    """
    dic = defaultdict(set, deepcopy(sdics[0])) if sdics else {}
    for d in sdics[1:]:
        for k, s in d.items():
            dic[k] |= s
    return dict(dic)


def setify_dict_values(dic: dict[Any, Any]) -> dict[Any, set[Any]]:
    """
    For a given `dic` turns all its values into sets:
    - if value `v` is Iterable but not string then it's turned into `set(v)`
    - if it's str or any non-Iterable then it's turned into `{v}`.
    """
    return {k: set(v) if iterable_not_str(v) else {v} for k, v in dic.items()}


def dict_set_union_force(*sdics: dict[Any, Iterable[Any] | Any]) -> dict[Any, set[Any]]:
    """
    This is "_force" version of `dict_set_union()`
    where for each key the value is turned into set if it's not set yet.
    More flexible although slower.

    *sdics: dictionaries;

    Result is a dictionary with union of all keys from each of dictionaries
    where for each key the value is a set being set-union of all iterables (except strings)
    under this key across all the dictionaries in `sdics`.
    If the value is 1-element of simple type or str then it is turned to 1-element set.
    If the value is str, say "abc", then it's turned into `{"abc"}`.
    If the value is None then it is turned into `{None}`.

    dic1 = {"a": {1, 2}, "b": {0, "0"}, "c": set(), "d": "ryq", "e": 1}
    dic2 = {"a": {3, 5, 7}, "b": None, "c": {0, 2}, "d": {"qq"}}
    dic3 = {"a": {0, 2, 5, 9}, "b": {0}, "e": 2}

    dict_set_union(dic1, dic2, dic3)  # ! TypeError: unsupported operand type(s) for |=: 'set' and 'NoneType'

    dict_set_union_force(dic1, dic2, dic3)
    # {'a': {0, 1, 2, 3, 5, 7, 9},
    #  'b': {0, '0', None},
    #  'c': {0, 2},
    #  'd': {'ryq', 'qq'},
    #  'e': {1, 2}

    dict_set_union_force(dic1)  # {'a': {1, 2}, 'b': {0, '0'}, 'c': set(), 'd': {'ryq'}, 'e': {1}}
    dict_set_union_force()      # {}
    """
    return dict_set_union(*list(map(setify_dict_values, sdics)))


# %%
def dict_list_sum(*ldics: dict[Any, list[Any]]) -> dict[Any, list[Any]]:
    """
    *ldics: dictionaries of lists;
        it's assumed that for each key the value is a list – otherwise TypeError is raised;
        use `dict_list_sum_force()` if not all values are lists.

    Result is a dictionary with union of all keys from each of dictionaries
    where for each key the value is a list being sum of all lists under this key
    across all the dictionaries in `ldics`.

    Examples
    --------
    dic1 = {"a": [1, 2], "b": [0, "0"], "c": [], "d": ["ryq"], "e": [1]}
    dic2 = {"a": [3, 5, 7], "c": [0, 2], "d": ["qq"], "b": [None]}
    dic3 = {"a": [0, 2, 5, 9], "b": [0], "e": [2]}

    dict_list_sum(dic1, dic2, dic3)
    # {'a': [1, 2, 3, 5, 7, 0, 2, 5, 9],
    #  'b': [0, '0', None, 0],
    #  'c': [0, 2],
    #  'd': ['ryq', 'qq'],
    #  'e': [1, 2]}
    """
    dic = defaultdict(list, deepcopy(ldics[0])) if ldics else {}
    for l in ldics[1:]:
        for k, s in l.items():
            dic[k] += s
    return dict(dic)


def listify_dict_values(dic: dict[Any, Any]) -> dict[Any, list[Any]]:
    """
    For a given `dic` turns all its values into lists:
    - if value `v` is Iterable but not string then it's turned into `list(v)`
    - if it's str or any non-Iterable then it's turned into `[v]`.
    """
    return {k: list(v) if iterable_not_str(v) else [v] for k, v in dic.items()}


def dict_list_sum_force(*ldics: dict[Any, Iterable[Any] | Any]) -> dict[Any, list]:
    """
    *ldics: dictionaries of lists;
        any value other then list is treated as 1-element list;

    Result is a dictionary with union of all keys from each of dictionaries
    where for each key the value is a list being list-sum of all lists under this key
    across all the dictionaries in `ldics`.

    dic1 = {"a": [1, 2], "b": (0, "0"), "c": [], "d": "ryq", "e": 1}
    dic2 = {"a": [3, 5, 7], "b": None, "c": {0, 2}, "d": ["qq"]}
    dic3 = {"a": (0, 2, 5, 9), "b": {0}, "e": 2}

    dict_list_sum(dic1, dic2, dic3)     # TypeError: can only concatenate tuple (not "NoneType") to tuple
    dict_list_sum_force(dic1, dic2, dic3)
    # {'a': [1, 2, 3, 5, 7, 0, 2, 5, 9],
    #  'b': [0, '0', None, 0]
    #  'c': [0, 2],
    #  'd': ['ryq', 'qq'],
    #  'e': [1, 2]}

    dict_list_sum_force(dic1)   # {'a': [1, 2], 'b': [0, '0'], 'c': [], 'd': ['ryq'], 'e': [1]}
    dict_list_sum_force()       # {}
    """
    return dict_list_sum(*list(map(listify_dict_values, ldics)))


# %%
