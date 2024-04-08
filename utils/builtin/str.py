#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: str utils
version: 1.0
type: code
keywords: [str, ]
description: |
    string utils
remarks:
    - In this file we try hard to not use any additional package - only built-ins!
      With few exceptions like itertools or functools (which)
todo:
sources:
file:
    date: 2019-11-20
    authors:
        - nick: arek
"""

#%%
def filter_str(string, iterable):
    res = list(filter(lambda s: s.find(string) > -1, iterable))
    return res


import re
def filter_re(regexp, iterable):
    reg = re.compile(regexp)
    res = list(filter(lambda s: reg.search(s), iterable))
    return res


def iterprint(iterable, sep="\n", pref="", suff="", strip=True):
    if strip:
        if isinstance(strip, str):
            iterable = [s.strip(strip) for s in iterable]
        else:
            iterable = [s.strip() for s in iterable]
    if pref or suff:
        iterable = [f"{pref}{s}{suff}" for s in iterable]
    print(sep.join(iterable))


iter_print = iterprint


""" Examples
ll = ["gura", "qqra", "burak", "qrak", "chmura"]
filter_str("qra", ll)

filter_re("qra", ll)
filter_re("^qra", ll)
filter_re("urak*$", ll)

iprint(ll)
iprint(ll, sep=" & ")
iprint(ll, pref="i.")
iprint(ll, suff=" ?")
iprint(ll, pref="i.", suff="!!!")
iprint(ll, " | ", pref="i.", suff=".~")

"""
#%%
