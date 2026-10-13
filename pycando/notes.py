#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:55:37 2026

@author: arek
"""

import numpy as np
ll = np.random.randint(-1, 9, size=22)
ll
list(ll > 0).index(True)
list(ll > 0).index(False)
list(ll > 3).index(True)

list(ll > 9).index(True)


def custom_index(it, f, default=-1):
    return next((i for i, e in enumerate(it) if f(e)), default)

def first_ok_idx(it, f, default=-1):
    try:
        idx = list(f(e) for e in it).index(True)
    except ValueError:
        idx = default
    return idx

def f(e): return e < 0

%timeit custom_index(ll, f)
# 3.84 μs ± 71.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit first_ok_idx(ll, f)
# 28.1 μs ± 920 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
