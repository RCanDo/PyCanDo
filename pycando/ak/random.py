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
    Convenience functions an utilities used in everyday work.
remarks:
todo:
sources:
file:
    date: 2020-09-13
    authors:
        - nick: arek
"""

#%%
import hashlib, time
import numpy as np
from .nppd import prob

from collections.abc import Sequence
from typing import TypeVar, Any
from numpy.typing import NDArray

#%%

def htr(max: int = 100) -> int:
    """
    Random integer with univariate distribution on [0, max)
    obtained by hashing (ssh224) current time and then taking modulo `max`.

    Examples
    --------
    htr(10)
    htr(1e7)

    from collections import Counter
    counter = Counter([htr(10) for t in range(2222)])
    counter  # Counter({3: 235, 8: 235, 9: 228, 7: 226, 4: 225, 6: 224, 5: 219, 1: 217, 2: 209, 0: 204})

    Aliases
    -------
    htrandint(max)
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

S = TypeVar("S", bound=Any)


def randkasp1(
        a: Sequence[S] | int,
        size: list | tuple | int | None = None,
        replace: bool = True,
        power: int = 2,
) -> S | NDArray[S] | int:
    """
    Choosing from Iterable `a` or, if `a` is int, from set {0, 1, ..., a-1}
    with distribution `[N, N - 1, ..., 1] ** pow` (normalised)
    where `N` is `len(a)` (if Iterabele) or just `a` (if int).
    == np.random.choice(a, size, replace, p=ak.p(np.arange(len(a), 0, -1) ** power))
    see help(np.random.choice).

    randkasp1(list('abcde'))    # np.str_('b')
    randkasp1(list('abcde'), size=(11, 11))

    import matplotlib.pyplot as plt
    M = 1000
    N = 100
    plt.scatter(x=range(M), y=ak.randkasp1(N, size=M, power=1), s=10)
    plt.scatter(x=range(M), y=ak.randkasp1(N, size=M, power=3), s=10)
    plt.scatter(x=range(M), y=ak.randkasp1(N, size=M, power=5), s=10)
    """
    if isinstance(a, (int, float)):
        N = a = int(a)
    else:
        N = len(a)
    return np.random.choice(a, size, replace, p=prob(np.arange(N, 0, -1) ** power))  # type: ignore  [return-value]

#%%
