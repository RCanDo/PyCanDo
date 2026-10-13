#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description:
    1. Exponential weights.
    2. Effectively applied Horner schema for generating exponential series.
remarks:
    - See file `power_series_horner.py` for analysis on
      "How to make generating power series fast – Horner schema applied effectively".
file:
    created: 2020.05.08
    authors:
        - nick: arek
"""
# %%
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray


@lru_cache
def power_series(p: float, n: int) -> tuple[float, ...]:
    """
    Very fast way of generating series (1, p, p^2, ..., p^(n)).
    Horner schema by recurrence:
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


def power_weights(p: float, n: int, reverse: bool = False) -> NDArray:
    """
    Normalised exponential weights of base `p`.
    `n` is the length of the final series `[1, p, ..., p**(n-1)] / sum(.)`

    Example
    -------
    ww = weights(.5, 10)  # array([0.50048876, 0.25024438, 0.12512219, 0.06256109, 0.03128055,
                          #        0.01564027, 0.00782014, 0.00391007, 0.00195503, 0.00097752])
    sum(ww)  # 1
    """
    if n < 2:
        raise Exception('n must be >= 2')
    if p >= 1:
        ww = np.array([1.] * n)
    elif p <= 0:
        raise Exception('p must be > 0')
    else:
        ww = np.array(power_series(p, n - 1))
    # normalisation
    ww /= ww.sum()   # more numerically accurate (stable)
    # ww *= (1 - p) / (1 - p * ww[-1])  # theoretically better (faster)
    if reverse:
        ww = ww[::-1]
    return ww


"""
Examples
--------
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

x = np.arange(10)
y = np.random.randn(10) + x
x
y
plt.plot(np.arange(10), y)

linreg.fit(x.reshape(-1, 1), y)
yhat0 = linreg.predict(x)  # ! ValueError: Expected 2D array, got 1D array instead...
                           # Reshape your data either using array.reshape(-1, 1) if your data has a single feature
                           # or array.reshape(1, -1) if it contains a single sample.
yhat0 = linreg.predict(x.reshape(-1, 1))
plt.plot(x, yhat0, label='no weights')

linreg.fit(x.reshape(-1, 1), y, sample_weight=weights(.5, 10))
yhat1 = linreg.predict(x.reshape(-1, 1))
plt.plot(x, yhat1, label='weights')

linreg.fit(x.reshape(-1, 1), y, sample_weight=weights(.5, 10, True))
yhat1 = linreg.predict(x.reshape(-1, 1))
plt.plot(x, yhat1, label='weights reversed')

plt.legend()
plt.tight_layout()

"""
