#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
title: pretty printing tools


file:
    authors:
        - nick: arek
          mail: rcando@int.pl

"""

# import io

#%%
def indent(obj, ind = "    ", rep=1, to_str=False):
    """"""
    ind *= rep
    reprobj = repr(obj).split("\n")
    reprobj = "\n".join(ind + s for s in reprobj)
    if to_str:
        return reprobj
    else:
        print(reprobj)

#%%
def iprint(obj, ind = "    ", to_str=False):
    """"""

    reprobj = indent(obj, ind, rep=1, to_str=True)

    if to_str:
        return reprobj
    else:
        print(reprobj)

#%%
