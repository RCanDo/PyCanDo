#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Convieniences And Utilities
subtitle: Based only on built-ins and basic libraries.
version: 1.0
type: module
keywords: [flatten, coalesce, ]
description: |
    Convieniens functions and utilities used in everyday work.
remarks:
    - We use only basic packages from standard library
      like functools, itertools, typing, time, math
todo:
sources:
file:
    usage:
        interactive: True
        terminal: False
    name: builtin.py
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

import time
# import common.builtin.timer as t

from functools import reduce, wraps
import itertools as it

import math as m


# %%
def coalesce(*args):
    """
    As in PostgreSQL: returns first not None argument;
    if all arguments are None then returns None.
    """
    ll = list(filter(lambda x: x is not None, args))
    ll.append(None)     # in case ll is empty
    return ll[0]


# %%
def dict_default(dic: dict, field: str, default: Any) -> Any:
    """defult value from dict field if it exists but is empty
    dict_default(dic, field, default) == coalesce(dic.get(field), default)
    but is faster then coalesce(...) as it does not use filter
    """
    value = dic.get(field, default)
    value = value if value is not None else default  # == coalesce(value, default)
    return value


"""
dic = dict(a=1, b=None)
%timeit dict_default(dic, 'b', 0)
# 103 ns ± 1.07 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
%timeit coalesce(dic.get('b'), 0)
# 400 ns ± 4.02 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

%timeit dict_default(dic, 'c', 2)
# 105 ns ± 0.583 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each
%timeit coalesce(dic.get('c'), 2)
# 399 ns ± 0.991 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
"""


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
    while not list(path.glob(pattern)):
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
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = replace_deep(v, what, to)
    elif isinstance(obj, list):
        obj = [replace_deep(l, what, to) for l in obj]
    else:
        obj = to if obj in what else obj
    return obj


# %%
def dict_depth(dic: dict):
    """
    `dict` is tree-like with all leaves being simple types (not collections),
    e.g. str, numeric or date.
    This procedure checks the depth of this tree
    BUT only at the 1st-1st-...-1st branch,
    what stems from assumtion that the tree is uniform
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
    rounding numbers to only r significant digits;
    if not number returns value unchanged;
    iterates over elements of Iterables (recursively) except strings which are ignored;
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
def lengthen(x: Any, n: int, as_list: bool = True) -> Union[list, tuple]:
    """Lengthening ll to the len(ll) == n;
    if ll is not iterable then it is turned into 1 element list (if `as_list` is True)
    or tuple (in other case) and then 'multiplied' n times.
    if len(ll) > n then ll is in fact shortened to have length == n.
    `as_list` is irrelevant in case of x is iterable.
    Examples:
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
    """better (?) version of lengthen0()
    Examples:
    Examples:
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
    Examples:
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
    Examples:
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
