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
    -
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

def htrandint(max=100):
    N = int(max)
    hashtime = hashlib.sha224(str(time.time()).encode('ascii')).hexdigest()
    return int(hashtime, 16) % N

"""
htrandint(10)
htrandint(1e7)
"""
# aliases
hashtimerandom = htrandint
hashtimerandint = htrandint

#%%