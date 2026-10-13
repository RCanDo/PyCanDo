#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
title: How to make generating power series fast – Horner schema applied effectively.
description:
    How to create pawer series fast.
conclusions:
    1. Proper usage of Horner schema i.e. recurrence with @lru_cache is the fastest solution.
    2. However, using Horner schema is not straitforward to make it effective.
       Simple list comprehension is faster.
    3. Using @lru_cache is tricky – connot use mutables inside and on return
       as the result from previous call will be returned in some circumstances.
    4. When using @lru_cache only immutables should be returned, i.e. tuple instead of list.
remarks:
    -
file:
    created: 2025.11.10
    authors:
        - nick: arek
"""
# %%
import numpy as np
from functools import reduce, lru_cache


@lru_cache
def power_series(p: float, n: int) -> tuple[float, ...]:
    """
    This is the fastest way of generating series (1, p, p^2, ..., p^(n)).
    Horner schema by recurrence with @lru_cache:
    power_series(.9, 2)   # (1, 0.9, 0.81)   powers  0, 1, 2
    power_series(.9, 6)   # (1, .9, .81, .729, .6561, .59049, .531441)   powers  0, 1, ..., 6
    power_series(.9, 1)   # (1, .9)
    power_series(.9, 0)   # (1,)
    power_series(.9, -1)  # (1,)
    """
    if n > 0:
        ser = power_series(p, n - 1)
        ser = *ser, ser[-1] * p
    else:
        ser = 1,
    return ser


powser = power_series  # alias


#%%
def powser_lc(p, n):
    """
    power series
    directly by list comprehension:
    powser_lc(.9, 6)  # [1, .9, .81, .729, .6561, .59049, .531441]
    """
    ser = [p ** k for k in range(n + 1)]
    return ser


# %%
@lru_cache
def pow(p, n):
    if n == 0:
        res = 1
    else:
        res = pow(p, n - 1) * p
    return res


def powser_lru(p, n):
    """
    directly by list comprehension:
    powser_lru(.9, 6)  # [1, .9, .81, .729, .6561, .59049, .531441]
    """
    ser = [pow(p, k) for k in range(n + 1)]
    # ???
    # ssum = 1/sum(ser)   # more accurate result   # (1-p)/(1 - p*ser[-1]) # theoretically better
    # ser = [s*ssum for s in ser]
    return ser


# %%
def powser_red(p: float, n: int) -> list[float]:
    """
    Horner schema by recurrence by reduce:

    help(reduce)
    reduce(lambda x, y: x + [y], [1, 2, 3], [1])
    powser_red(.9, 6)  # [1, .9, .81, .729, .6561, .59049, .531441]
    """
    ser = reduce(lambda x, y: x + [x[-1] * y], [p] * (n - 1), [1, p])
    # ???
    # ssum = 1/sum(ser)   # more accurate result   # (1-p)/(1 - p*ser[-1]) # theoretically better
    # ser = [s*ssum for s in ser]
    return ser


def powser_rec(p: float, n: int) -> list[float]:
    """
    Horner schema by recurrence:
    powser_rec(.9, 6)  # [1, .9, .81, .729, .6561, .59049, .531441]
    """
    if n > 1:
        ser = powser_rec(p, n - 1)
        ser.append(p * ser[-1])
    else:
        ser = [1, p]
    return ser


#%%
"""
Examples
--------

%timeit powser_lc(.9, 100)       # 6.62 μs ± 53.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit powser_lru(.9, 100)      # 10.4 μs ± 40 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit powser_red(.9, 100)      # 23.9 μs ± 160 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
%timeit powser_rec(.9, 100)      # 14.6 μs ± 287 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit power_series(.9, 100)  # 86.1 _ns_ ± 0.494 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
                                 # the winner – about 7 times faster then list comprehension

# Conclusion
It's better to multiply again then remember result in the list and reuse it.
"""
# %%
"""
Strange things happen when using @lru_cache with mutable like `list(.)`:
"""
@lru_cache
def powser_rec_lru_m(p: float, n: int) -> list[float]:
    if n > 1:
        ser = powser_rec_lru_m(p, n - 1)
        ser.append(p * ser[-1])
    else:
        ser = [1, p]
    return ser
"""
powser_rec_lru_m(.9, 5)  # [1, .9, .81, .729, .6561, .59049]
powser_rec_lru_m(.9, 9)  # [1, .9, .81,  ...,  .4304, .3874]  length 10  ok
powser_rec_lru_m(.9, 5)  # [1, .9, .81,  ...,  .4304, .3874]  length 10  very wrong! – remebers state from previous call
# it does not happen without lru_cache !
# but then is much slower

# while with  lru_cache  it's the fastest method
%timeit powser_rec_lru_m(.9, 100)  # 84.7 _ns_ ± 0.54 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
"""
# %%
"""
However simple change from list to tuple fixes the problem
"""
@lru_cache
def powser_rec_lru_m2(p: float, n: int) -> tuple[float, ...]:
    if n > 1:
        ser = list(powser_rec_lru_m2(p, n - 1))
        ser.append(p * ser[-1])
    else:
        ser = [1, p]
    return tuple(ser)   # ! tuple()
"""
powser_rec_lru_m2(.9, 5)  # [1, .9, .81, .729, .6561, .59049]
powser_rec_lru_m2(.9, 9)  # [1, .9, .81,  ...,  .4304, .3874]
powser_rec_lru_m2(.9, 5)  # [1, .9, .81, .729, .6561, .59049]  ok
"""

#%%
"""
what about summation ?
"""
def powsum_lc(p, n):
    """
    powsum_lc(.9, 6)  # 5.217031
    """
    return sum(p ** k for k in range(n + 1))


def powsum_lru(p, n):
    """
    directly by list comprehension:
    powsum_lru(.9, 6)  # 5.217031
    """
    return sum(pow(p, k) for k in range(n + 1))


def powsum_rec(p: float, n: int) -> tuple[float, ...]:
    """
    Horner schema by recurrence:
    powsum_rec(.9, 6)  # (5.217031, 0.531441)
    """
    if n > 0:
        s, s_ = powsum_rec(p, n - 1)
        s_n = p * s_
        s += s_n
    else:
        s, s_n = 1, 1
    return s, s_n


@lru_cache
def powsum_rec_lru(p: float, n: int) -> tuple[float, ...]:
    """
    Horner schema by recurrence:
    powsum_rec_lru(.9, 6)  # (5.217031, 0.531441)
    """
    if n > 0:
        s, s_ = powsum_rec_lru(p, n - 1)
        s_n = s_ * p
        s += s_n
    else:
        s, s_n = 1, 1
    return s, s_n

"""
%timeit powsum_lc(.9, 100)  # 7.99 μs ± 124 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit powsum_lru(.9, 100)  # 12.5 μs ± 749 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit powsum_rec(.9, 100)  # 15.2 μs ± 90.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit powsum_rec_lru(.9, 100)  # 88.9  ns  ± 1.54 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

By the way: when returning just a sum there is no problem with returnig previous state:
powsum_rec_lru(.9, 6)  # (5.217031, 0.531441)
powsum_rec_lru(.9, 9)  # (6.51321,  0.38742)
powsum_rec_lru(.9, 6)  # (5.217031, 0.531441)   ok
"""
# %%
"""
clue of the problem – using mutable (list insted of tuple)
"""
def f():
    x = [0, 1, 2]
    return x

a = f()
a.append(3)
a               # [0, 1, 2, 3]
b = f()
b               # [0, 1, 2]

@lru_cache
def fc():
    x = [0, 1, 2]
    return x

a = fc()
a.append(3)
a               # [0, 1, 2, 3]
b = fc()
b               # [0, 1, 2, 3]

# no problems with tuple
@lru_cache
def ftc():
    x = (0, 1, 2)
    return x

a = ftc()
a.append(3)     # ! AttributeError: 'tuple' object has no attribute 'append' – it's immutable !
a = *a, 3
a               # (0, 1, 2, 3)
b = ftc()
b               # (0, 1, 2)    ok

# %%
"""
Fibonacci series (not only n-th element)
"""
def fib(n):
    if n < 2:
        res = [1, 1]
    else:
        res = fib(n - 1)
        res.append(res[-1] + res[-2])
    return res

fib(9)  # [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
fib(3)  # [1, 1, 2, 3]


# make it faster
@lru_cache
def fib_lru(n):
    if n < 2:
        res = [1, 1]
    else:
        res = fib_lru(n - 1)
        res.append(res[-1] + res[-2])
    return res

fib_lru(3)   # [1, 1, 2, 3]
fib_lru(9)   # [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
fib_lru(3)   # [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]    ! oops!  previous state returned :(
fib_lru(11)  # [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
fib_lru(9)   # [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]   ! oops!  previous state returned :(


# fix it by replacing lust with tuple
@lru_cache
def fib_lru_t(n):
    if n < 2:
        res = (1, 1)
    else:
        res = fib_lru_t(n - 1)
        res = *res, res[-1] + res[-2]
    return res

fib_lru_t(3)   # (1, 1, 2, 3)
fib_lru_t(9)   # (1, 1, 2, 3, 5, 8, 13, 21, 34, 55)
fib_lru_t(3)   # (1, 1, 2, 3)    OK!
fib_lru_t(11)  # (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)
fib_lru_t(9)   # (1, 1, 2, 3, 5, 8, 13, 21, 34, 55)    OK!

"""
%timeit fib(100)        # 9.49  μs  ± 151 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit fib_lru_t(100)  # 50.1 _ns_ ± 0.464 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
"""
