#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Transformations
version: 1.0
type: module
keywords: [transformation, function, mapping]
description: |
    Variables transformation helper functions
remarks:
todo:
sources:
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2022-01-12
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import numpy as np
import pandas as pd

from functools import partial, wraps, update_wrapper


# %%
def recur(fun):

    @wraps(fun)
    def recurrent_version(ss, r=1, *args, **kwargs):
        if r < 1:
            return ss
        else:
            return recurrent_version(fun(ss, *args, **kwargs), r - 1, *args, **kwargs)

    return recurrent_version


def split_pos_neg_0(fun):

    @wraps(fun)
    def split_version(ss, *args, **kwargs):
        ss = pd.Series(ss)
        ss_neg = ss[ss < 0]
        ss_0 = ss[ss == 0]
        ss_pos = ss[ss > 0]
        ss = pd.concat([-fun(-ss_neg, *args, **kwargs), ss_0, fun(ss_pos, *args, **kwargs)], axis=0)[ss.index]
        return ss

    return split_version


# %%
def log1(ss, c=1):
    return np.log(ss + c)


# %%
def test0(ss, c):
    return 2 * ss + c


"""
rectest = recur(test0)
rectest(np.arange(3), 1, c=1)
rectest(np.arange(3), 1, 1)
rectest(np.arange(3), 1, 2)
rectest(np.arange(3), 2, 1)         #!!!
"""


# %%
@recur
def rectest(ss, c=1):
    return 2 * ss + c


"""
rectest(np.arange(3), 1, c=1)
rectest(np.arange(3), 1, 1)
rectest(np.arange(3), 1, 2)
rectest(np.arange(3), 2, 1)
"""


# %%
@recur
def rlog1(ss):
    return np.log(ss + 1)


rlog1.__name__ = "rlog1"


@recur
def rexp1(ss):
    return np.exp(ss) - 1


rexp1.__name__ = "rexp1"


"""
rlog1(np.arange(3))

rlog1(np.arange(3), 1)
rexp1(rlog1(np.arange(3), 1))

rlog1(np.arange(3), 2)
rexp1(rlog1(np.arange(3), 2), 2)
"""


# %%
@split_pos_neg_0
@recur
def srlog1(ss):
    return np.log(ss + 1)


srlog1.__name__ = "srlog1"


@split_pos_neg_0
@recur
def srexp1(ss):
    return np.exp(ss) - 1


srexp1.__name__ = "srexp1"


"""
ss = pd.Series(np.arange(-3, 3))
srlog1(ss)
srlog1(ss, 2)

srexp1(srlog1(ss))

srexp1(srlog1(ss, 1))

srexp1(srlog1(ss, 2), 2)

from functools import partial
trans2 = partial(srlog1, r=2)
trans2(ss)

"""

# %%
tlog1 = partial(srlog1, r=1)
update_wrapper(tlog1, srlog1)
tlog1.__name__ = "tlog1"

tlog2 = partial(srlog1, r=2)
update_wrapper(tlog2, srlog1)
tlog2.__name__ = "tlog2"

texp1 = partial(srexp1, r=1)
update_wrapper(texp1, srexp1)
texp1.__name__ = "texp1"

texp2 = partial(srexp1, r=2)
update_wrapper(texp2, srexp1)
texp2.__name__ = "texp2"

# %%
