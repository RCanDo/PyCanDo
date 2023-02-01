#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:01:45 2022

@author: arek

It's example file NOT a part of module!
"""

# %%
# %%
# app/xyz.py
import argparse

INFO= """..."""

def get_parser():

    parser = argparse.ArgumentParser(
        description = INFO,
        formatter_class = lambda prog: argparse.HelpFormatter(
            prog, max_help_position=60, width=110)
    )

    parser.add_argument("--arg", default=None, type=int, help="...")
    parser.add_argument("--arg2", default=None, type=str, help="...")

    return parser


def get_args():
    parser = get_parser()
    print(INFO)

    args = parser.parse_args()

    # args processing -- should be wrapped in as function
    args.new = 1
    args.new2 = f"{args.arg}"
    # ...

    return args


def main(args):
    result = []
    # result.append(f(args.arg))
    # result.append(g(args.arg2, arg.new))
    # result.append(h(arg.new2))
    # ...
    return result


#%%
if __name__ == '__main__':
    args = get_args()
    main(args)


# %%
# %%
# common/testutils/testutils.py
def parse_with(parser, params, *flags):
    """
    params = {k: v, k1: v0, ...}
    flags = ["k", "k0", ...]
    """
    res = []
    for k, v in params.items():
        res.append(f'--{k}')
        res.append(f'{str(v)}')
    for k in flags:
        res.append(f'--{k}')
    return parser.parse_args(res)


# %%
# %%
# tests/test_xyz.y
from app.xyz import main, get_parser
from common.testutils import parse_with

def test_xyz():
    parser = get_parser()
    args = parse_with(parser,
               {"arg": 1,
                "arg2": "qq"},
               )

    # args processing -- should be wrapped in as function
    args.new = 1
    args.new2 = f"{args.arg}"
    # ...

    main(args)

    # ...
