#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Convieniences And Utilities
subtitle: Based only on built-ins.
version: 1.0
type: code
keywords: [flatten]   # there are always some keywords!
description: |
    Convieniens functions an utilities used in everyday work.
content:
    - flatten(lst:list) -> list  -- deep flatten a list
remarks:
    - In this file we not use any additional package - only built-ins!
todo:
sources:
file:
    usage:
        interactive: True
        terminal: True
    name: builtin.py
    path: D:/ROBOCZY/Python/RCanDo/ak/
    date: 2019-11-20
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

#%%
"""
pwd
cd D:/ROBOCZY/Python/RCanDo/...
ls
"""

from functools import reduce
from typing import List, Tuple, Dict, Set, Optional, Union, NewType, Iterator, Any

#%%
def coalesce(*args):
    ll = list(filter(lambda x: not x is None, args))
    ll.append(None)
    return ll[0]

#%%
import time

def timeit(fun):
    """decorator for timing"""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fun(*args, **kwargs)
        t1 = time.time()
        dt = t1 - t0
        print("Execution time: {:3.5f}".format(dt))
        return result
    return wrapper

#%%
def lengthen0(x, n, as_list=True):
    '''Lengthening ll to the len(ll) == n;
    if ll is not iterable then it is turned into 1 element list (if `as_list` is True)
    or tuple (in other case) and then 'multiplied' n times.
    if len(ll) > n then ll is in fact shortened to have length == n.
    `as_list` is irrelevant in case of x is iterable.

    '''

    if isinstance(x, (list, tuple)):
        is_list = isinstance(x,list)
        l = len(x)
        if l < n:
            if is_list:
                x = x + [x[-1]]*(n-l)
            else:
                x = x + (x[-1],)*(n-l)
        else:
            x = x[:n]
    elif isinstance(x, (int, float, str)):
        if as_list:
            x = [x]*n
        else:
            x = (x,)*n
    else:
        print("Error: `x` must be in (int, float, str) or iterable of these.")
        sys.exit(1)

    return x

#%%
from itertools import *
def lengthen(iterable: Iterator, n: int, as_list: bool=True) -> Iterator:
    """better version of lengthen0()"""
    iterable = islice(chain(iterable, repeat(iterable[-1], max(0, n - len(iterable)))), n)

    if as_list:
        result = list(iterable)
    else:
        result = tuple(iterable)

    return result

# lengthen0([1, 3, 2, 5], 9, True)

#%%
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

#%%
def paste(left, right,
          by_left: bool = True, sep: str = "", flat: bool = True):
    """
    Works similar to R's paste().
    by_left: bool
    Examples:
    paste(['a', 'b'], [1, 2]) == ['a1', 'a2', 'b1', 'b2']
    paste(['a', 'b'], [1, 2], False) == ['a1', 'b1', 'a2', 'b2']
    paste(['a', 'b'], [1, 2], sep="_") == ['a_1', 'a_2', 'b_1', 'b_2']
    """
    result = []

    if not isinstance(left, list) and isinstance(left, (range, tuple, set, dict)):
        left = list(left)
    if not isinstance(right, list) and isinstance(right, (range, tuple, set, dict)):
        right = list(right)

    def l_sep_r(l, r):
        return "{}{}{}".format(l, sep, r)

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


#%%
def dict_set_union(dic1: Dict[Any, Set], dic2: Dict[Any, Set]) -> Dict[Any, Set]:
    """
    dict1, dict2: dictionaries of sets !!!
    Returns dictionary with all the keys from both dicts
    and union of sets for each key.

    Examples:
    dic1 = {"a":{1, 2}, "b":{0, "0"}, "c":set()}
    dic2 = {"a":{3, 5, 7}, "c":{0, 2}, "d":{"qq"}}

    dict_set_union(dic1, dic2)

    union(*dic2.values())
    union(*dic2.values(), *dic1.values())
    """
    keys = set(dic1.keys()).union(dic2.keys())

    dic_new = dict()
    for k in keys:
        dic_new[k] = dic1.get(k, set()).union(dic2.get(k, set()))

    return dic_new

def union(*args):
    return reduce(lambda a, b: a.union(b), args, set())


#%%
