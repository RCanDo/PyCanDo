#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 07:40:17 2020

@author: arek
"""

import numpy as np
from functools import reduce

#%%
#%%
def powseries(x, n):
    #print(n)
    if n > 1:
        ser = powseries(x, n-1)
        ser.append(x*ser[-1])
    else:
        ser = [1, x]
    #print(ser)
    return ser

powseries(.9, 6)

#%%
help(reduce)

reduce(lambda x, y: x + [y], [1, 2, 3], [1])
#%%
def powser(p, n):
    ser = reduce(lambda x, y: x + [x[-1]*y], [p]*(n - 1), [1, p])
    ssum = 1/sum(ser)   # more accurate result   # (1-p)/(1 - p*ser[-1]) # theoretically better
    ser = [s*ssum for s in ser]
    #print(sum(ser))
    return ser
   
powser(.9, 6)

#%%
def fun(x, y):
    return x + [x[-1]*y]

def powser2(p, n):
    ser = reduce(fun, [p]*(n - 1), [1, p])
    ssum = 1/sum(ser)   # more accurate result   # (1-p)/(1 - p*ser[-1]) # theoretically better
    ser = [s*ssum for s in ser]
    #print(sum(ser))
    return ser
   
powser(.9, 6)

#%% 
def powser_lc(p, n):
    ser = [p**k for k in range(n)]
    ssum = 1/sum(ser)   # more accurate result   # (1-p)/(1 - p*ser[-1]) # theoretically better
    ser = [s*ssum for s in ser]
    #print(sum(ser))
    return ser

#%%
%timeit powseries(.9, 100)  # 24 µs ± 57.2 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
%timeit powser(.9, 100)     # 37 µs ± 130 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
%timeit powser_lc(.9, 100)  # 16.6 µs ± 23.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


#%%
def f_red(p)
reduce(lambda x: x + [x[-1]], [p]*(n - 1), [p])



#%%
N=6
x=.9
weights = [(1-x)*x**k for k in range(N)]
weights
sum(weights)

#%%
def weights(p, n, reverse=False):
    # n is the length of the final series [1, p, ..., p**(n-1)]/sum
    #
    if n < 2:
        raise Exception('n must be >= 2')
    if p >= 1:
        ww = np.array([1.] * n)
    elif p <= 0:
        raise Exception('p must be > 0')
    else:
        ww = np.array(reduce(lambda x, y: x + [x[-1]*y], [p]*(n - 2), [1, p]))
    #ww /= ww.sum()   # more numerically accurate (stable)
    #ww *= (1-p)/(1 - p*ww[-1]) # theoretically better (faster)
    if reverse:
        ww = ww[::-1]
    return ww
        
ww = weights(.5, 10)
print(ww)
sum(ww[::-1])

#%%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

x = np.arange(10)
y = np.random.randn(10) + x
x 
y
plt.plot(np.arange(10), y)

linreg.fit(x.reshape(-1, 1), y)
yhat0 = linreg.predict(x.reshape(-1, 1))
plt.plot(x, yhat0, label='0')

linreg.fit(x.reshape(-1, 1), y, sample_weight=weights(.5, 10))
yhat1 = linreg.predict(x.reshape(-1, 1))
plt.plot(x, yhat1, label='1')

linreg.fit(x.reshape(-1, 1), y, sample_weight=weights(.5, 10, True))
yhat1 = linreg.predict(x.reshape(-1, 1))
plt.plot(x, yhat1, label='2')

plt.legend()

#%%