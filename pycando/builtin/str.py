#! python3
# -*- coding: utf-8 -*-
"""
---
title: str utils
version: 1.0
keywords: [str, ]
description: |
    string utils
todo:
sources:
file:
    date: 2019-11-20
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
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

iterprint(ll)
iterprint(ll, sep=" & ")
iterprint(ll, pref="i.")
iterprint(ll, suff=" ?")
iterprint(ll, pref="i.", suff="!!!")
iterprint(ll, " | ", pref="i.", suff=".~")

"""
#%%
