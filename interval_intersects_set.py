#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:31:52 2022

@author: arek
"""

def interval_intersects_set(interval, collection, sort=True, closed=(True, True), values=False):
    """
    DON'T USE IT - IT'S SLOW!!!
    interval: (a, b), a <= b
    collection: list/set of values
    sort: boolean
        if True then sorts `collection` in ascending order;
        turn it off (pass False) if `collections` already sorted (to spare time);
    REMEMBER that algorithm assumes `collection` to be ordered whith ascending order;
    so if `collection` is not ordered pass `sort=True` (default);
    or rather turn it off (pass False) only if you are shure `collection` is already sorted.
    """

    if len(collection) > 0:

        if sort:
            collection = sorted(collection)  # list
        else:
            collection = list(collection)

        if closed[1]:
            def is_too_big(v):
                return v > interval[1]
        else:
            def is_too_big(v):
                return v >= interval[1]

        # 1. removing values too large
        # assumed ascending order
        stop = False
        while not stop:
            if is_too_big(collection[-1]):
                collection.pop()
                stop = len(collection) == 0
            else:
                stop = True

    if len(collection) > 0:

        if closed[0]:
            def is_too_small(v):
                return v < interval[0]
        else:
            def is_too_small(v):
                return v <= interval[0]

        # 2. removing values too small

        # reversing order to descending as .pop() is faster then .remove() (in theory at least)
        collection.reverse()
        stop = False
        while not stop:
            if is_too_small(collection[-1]):
                collection.pop()
                stop = len(collection) == 0
            else:
                stop = False

        collection = collection.reverse()

    if values:
        return collection
    else:
        return len(collection) > 0
