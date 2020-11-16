# -*- coding: utf-8 -*-
#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: random utilities
subtitle:
version: 1.0
type: code
keywords: [random, time]
description: |
    Convieniens functions an utilities used in everyday work.
content:
    - htr(max: int=100) -> int
    - randkasp1(a: Union[np.array, int], 
              size: Union[List, int] = 1, 
              replace: bool = True, 
              power: int=2) -> Union[np.array, int]
remarks:
todo:
sources:
file:
    usage:
        interactive: True
        terminal: True
    name: random.py
    path: D:/ROBOCZY/Python/RCanDo/ak/
    date: 2020-09-13
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

from typing import List, Tuple, Dict, Set, Optional, Union, NewType

#%%
#%%
import hashlib, time

def htr(max: int=100) -> int:
    """
    Random integer with univariate distribution on [0, max)
    obtained by hashing (ssh224) current time and then taking modulo `max`.
      
    Examples
    --------
    > htrandint(10)
    > htrandint(1e7)
    
    Aliases
    -------
    hashtimerandom(max)
    hashtimerandint(max)
    """
    N = int(max)
    hashtime = hashlib.sha224(str(time.time()).encode('ascii')).hexdigest()
    return int(hashtime, 16) % N

# aliases
htrandint = htr
hashtimerandom = htr
hashtimerandint = htr

#%%
from .nppd import prob
import numpy as np

def randkasp1(a: Union[np.array, int], 
              size: Union[List, int] = None, 
              replace: bool = True, 
              power: int=2) -> Union[np.array, int]:
    """
    ==np.random.choice(a, size, replace, p=ak.p(np.arange(len(a), 0, -1)**power))
    see help(np.random.choice)
    
    import matplotlib.pyplot as plt
    M = 1000
    plt.scatter(x=range(M), y=[ak.randkasp1(N, power=3) for k in range(M)])
    """
    if isinstance(a, (int, float)):
        N = a = int(a)
    else:
        N = len(a)
    return np.random.choice(a, size, replace, p=prob(np.arange(N, 0, -1)**power))


def randkasp2(a: Union[np.array, int], 
              size: Union[List, int] = None, 
              replace: bool = True, 
              power: int=2) -> Union[np.array, int]:
    """
    ==np.random.choice(a, size, replace, p=ak.p(np.arange(len(a), 0, -1)**power))
    see help(np.random.choice)
    
    import matplotlib.pyplot as plt
    M = 1000
    plt.scatter(x=range(M), y=[ak.randkasp1(N, power=3) for k in range(M)])
    """
    if isinstance(a, (int, float)):
        N = a = int(a)
    else:
        N = len(a)
    return np.random.choice(a, size, replace, p=prob(np.arange(N, 0, -1)**power))

#%%