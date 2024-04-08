#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Applying conditions on data frame
version: 1.0
type: modeule
keywords: [filtering, selectinon, creating new columns, ...]
description: |
    Application of conditions on the variables from data frame:
    - filtering rows according to some conditions on variables,
    - creating new variables from other,
    - dropping and selecting columns.
    MultiIndex allowed.
    It is possible to refer to functions passed into condapply() as keword arg.
todo:
    - pass numbers and slices to select and drop
file:
    date: 2022-12-23
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
import pandas as pd
import utils.builtin as bi
import utils.data_utils as du


# %%
@bi.timeit
def condapply(df: pd.DataFrame, conditem: dict, levels_delim: str = "/", **globs) -> pd.DataFrame:
    """
    levels_delim : str = "/"
        in case of multiindex;
        in names of variables "a/b" means ('a', 'b') column;
        if None then there is no search for splitting colnames (i.e. no multiindex assumed);
    **globs:
        other objects (callables/funcions or iterabels/containers or simple types or whatever)
        passed from global envir;
        unfortunatelly referring to globals directly not always works (why?);
        e.g. in case of using functions from imported modules, like np.log()
        one must pass `np=np` and refer to it in a condlist item via `np.log(x)`
        or pass `log=np.log` and refer to it via `log(x)`;

    Examples
    --------
    import numpy as np
    np.random.seed(2)
    df = pd.DataFrame(np.random.randint(-3, 3, [33, 3]), columns=['col1', 'col2', 'col3'])
    df

    conditem = {
        "cols": {'x': 'col1', 'y': 'col2', 'z': 'col3'},  # ['col1', 'col2', 'col3']  <- 'a', 'b', 'c', ...
        "filter": ['x > y', 'z.isin([-1, 1])'],
        "create": {'new': 'x*y', 'new2': 'np.log(abs(y))'},
        #
        "cols.1": {"a": 'new2', 'b': 'col1'},
        'create.1': {'not_inf': 'a == -np.inf'},
        'filter.1': ['(b >= 0) & (a != -np.inf)'],
        #
        'cols.2': ['col2'],
        'create.2': {'is_neg_3': 'a == -3'}
        }

    dfc = condapply(df, conditem)
    dfc

    ## Multiindex
    np.random.seed(2)
    dfa = pd.DataFrame(np.random.randint(-3, 3, [33, 3]), columns=['col1', 'col2', 'col3'])
    dfb = pd.DataFrame(np.random.randint(-3, 3, [33, 3]), columns=['col1', 'col2', 'qq'])
    df = pd.concat({"A": dfa, "B": dfb}, axis=1)
    df

    conditem = {
        "cols": {'x': ['A', 'col1'], 'y': ['B','col2'], 'z': ['B','qq']},
                    # ['col1', 'col2', 'col3']  <- 'a', 'b', 'c', ...
        "filter": ['x > y', 'z.isin([-1, 1])'],
        "create": {'A/new': 'x*y', 'B/new2': 'np.log(abs(y))'},
        #
        "cols.1": {"a": 'B/new2', 'b': 'A/col1'},
        'create.1': {'C/is_inf': 'a == -np.inf'},
        'filter.1': ['(b >= 0) & (a != -np.inf)'],
        # #
        # 'cols.2': ['B/col2'],
        # 'create.2': {'is_neg_3': 'a == -3'}
        }


    dfc = condapply(df, conditem)
    dfc

    ## Objects from global env
    def fun(x):
        return x**2

    conditem = {
        "cols": {'x': 'A/col1', 'y': 'B/col2', 'z': 'B/qq'},  # ['col1', 'col2', 'col3']  <- 'a', 'b', 'c', ...
        "filter": ['x > y', 'z.isin([-1, 1])'],
        "create": {'A/new': 'x*y', 'B/new2': 'fun(y)'},
        }

    dfc = condapply(df, conditem)
    dfc
    """
    def proper_name(name):
        if not isinstance(name, str):  # in case of multindex
            name = tuple(name)
        else:
            name = tuple(name.split(levels_delim)) if levels_delim else name
        return name

    if globs:
        for k, v in globs.items():
            locals().__setitem__(k, v)

    print(df.shape)

    for key, group in conditem.items():
        print(key)

        if key.split(".")[0] in ('cols', 'columns', 'vars', 'variables'):
            cols = group
            if isinstance(cols, list):
                letters = list("abcdefghijklmnopqrstuvwxyz")[:len(cols)]
                cols = {letr: name for letr, name in zip(letters, cols)}
            for letr, name in cols.items():
                name = proper_name(name)
                print(" " * 3, f"{letr} = {name}")
                locals().__setitem__(letr, df[name])

        elif key.split(".")[0] == 'filter':
            filters = group
            for filtr in filters:
                print(" " * 3, filtr)
                idx = eval(filtr)
                if isinstance(idx, slice):
                    df = df.iloc[idx, :]
                elif isinstance(idx[0], (pd.np.bool_, bool,)):
                    df = df[idx]
                print(" " * 6, df.shape)

        elif key.split(".")[0] == 'create':
            news = group
            for var, cond in news.items():
                var = proper_name(var)
                print(" " * 3, f"{var} = {cond}")
                df[var] = eval(cond)

        # elif key.split(".")[0] == 'apply':
        #     # apply one function to many columns (and replace them with result)    ???

        elif key.split(".")[0] == 'select':
            cols = group  # assume this is list of names
            # ? what about list of numbers (positions) or slices ?
            cols = [proper_name(c) for c in cols]
            df = df[cols]

        elif key.split(".")[0] == 'drop':
            cols = group  # assume this is list of names
            cols = [proper_name(c) for c in cols]
            df = df.drop(cols, axis=1, errors="ignore")

        elif key.split(".")[0] in ('set_types', 'change_types'):
            col_types = group  # assume this is dictionary:
            # {'type': ['col1', 'col2', ...], 'type2': ['colx', 'coly', ...], ...}
            # indicating what type should be set to which columns;
            df = du.change_types(df, col_types)

        else:
            pass

    return df


# %%
def condlistapply(df: pd.DataFrame, condlist: list):
    """
    Examples
    --------
    np.random.seed(2)
    df = pd.DataFrame(np.random.randint(-3, 3, [33, 3]), columns=['col1', 'col2', 'col3'])
    df

    condlist = [
        {
        "cols": {'x': 'col1', 'y': 'col2', 'z': 'col3'},  # ['col1', 'col2', 'col3']  <- 'a', 'b', 'c', ...
        "filter": ['x > y', 'z.isin([-1, 1])'],
        "create": {'new': 'x*y', 'new2': 'np.log(abs(y))'},
        },
        #
        {
        "cols": {"a": 'new2', 'b': 'col1'},
        'create': {'not_inf': 'a == -np.inf'},
        'filter': ['(b >= 0) & (a != -np.inf)'],
        },
        #
        {
        'cols': ['col2'],
        'create': {'is_neg_3': 'a == -3'},
        }
        ]

    dfc = condlistapply(df, condlist)
    dfc
    """
    for conditem in condlist:
        df = condapply(df, conditem)
    return df


# %%
