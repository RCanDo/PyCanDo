#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Data preparation
project: Empirica
version: 1.0
type: module             # module, notebook, analysis, model, tutorial, help, example, ...
keywords: [preprocessing, loading, type corrections, selection, filtering]
description: |
    See README.md
remarks:
    - originally .../pipeline/prepare_data.py
todo:
    - problem 1
sources:
    - title:
      chapter:
      pages:
      link: https://the_page/../xxx.domain
      date:    # date of issue or last edition of the page
      authors:
          - nick:
            fullname:
            email:
      usage: |
file:
    usage:
        interactive: False   # if the file is intended to be run interactively e.g. in Spyder
        terminal: True      # if the file is intended to be run in a terminal
    date: 2022-01-12
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
        - nick: jerry
          fullname: Jeremiasz Leszkowicz
          email:
              - jeremiasz.leszkowicz@quantup.pl
"""
#%%
import sys
from typing import List, Tuple, Set
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#from utils.builtin import flatten, coalesce, dict_set_union, union
import utils.setup as su

#%%


#%%

def change_types(data: pd.DataFrame, col_types: dict = su.COL_TYPES) -> pd.DataFrame:
    """transforms each `data`s column type to the one registered in `col_types`

    Returns pd.DataFrame with 'proper' variables (columns) types.
    NOTICE:
    What these types should be need to be discovered for the new data version
    only after first load and investigation; see: .../analyses/new_data_insight.py
    VARS_REGISTERED already have proper types noticed (and applied) in c.COL_TYPES
    but new variables need to be investigated first and only then
    the VARS_NEW may be properly filled in.

    It's a loop over all registered variables.
    """
    if col_types.get('integer'):
        for col in col_types['integer']:
            try:
                data[col] = data[col].astype(int)
            except KeyError:
                print(f"There is no '{col}' in data.")
    if col_types.get('float'):
        for col in col_types['float']:
            try:
                data[col] = data[col].astype(str).str.replace(',', '.')
                data[col] = data[col].astype(float)
            except KeyError:
                print(f"There is no '{col}' in data.")
    if col_types.get('date'):
        for col in col_types['date']:
            try:
                data[col] = pd.to_datetime(data[col])
            except KeyError:
                print(f"There is no '{col}' in data.")
    if col_types.get('string'):
        for col in col_types['string']:
            try:
                data[col] = data[col].astype(str).replace({'nan': None})
            except KeyError:
                print(f"There is no '{col}' in data.")
    if col_types.get('category'):
        for col in col_types['category']:
            try:
                data[col] = data[col].astype('category')
            except KeyError:
                print(f"There is no '{col}' in data.")
    if col_types.get('ord-category'):
        for dict_col in col_types['ord-category']:
            for key, value in dict_col.items():
                try:
                    data[key] = pd.Categorical(data[key], categories=value, ordered=True)
                except KeyError:
                    print(f"There is no '{key}' in data.")
    return data


def variables(data: pd.DataFrame, verbose=True) -> Tuple[Set, Set, Set]:
    """
    When new version of data is loaded it's good to have
    comparison with current version of data on:
    - VARS_REGISTERED -- vairables in current version of data;
    - lost_variables i.e. not present in new version of data,
    - new_variables i.e. present in new version but absent in current version of data.
    """
    new_variables  = list(set(data.columns).difference(su.VARS_REGISTERED))
    lost_variables = list(set(su.VARS_REGISTERED).difference(data.columns))

    if verbose:
        print("Old variables:")
        print(su.VARS_REGISTERED)

        print("Lost variables:")
        print(lost_variables)

        print("New variables:")
        print(new_variables)

    return su.VARS_REGISTERED, lost_variables, new_variables


def validation(data: pd.DataFrame):
    """
    !!! THINK IT OVER CAREFULLY !!!
    What things should be checked up and what for???
    """
    assert data.shape[1] == len(su.VARS_REGISTERED), 'Check number of columns.'
    assert data.shape[0] == su.CASES, "Check number of rows."


def read_raw(path: str = su.PATH_DATA_CSV,
        sep: str = ';',
        encoding: str = 'utf-8'
        ) -> pd.DataFrame:
    """
    Reading RAW data from .csv file BEFORE ANY PREPROCESSING !
    """
    # read
    data = pd.read_csv(path, sep=sep, encoding=encoding)

    print("Data raw")
    print("--------")

    vars_old, vars_lost, vars_new = variables(data, verbose=True)

    print("Number of columns: {}".format(data.shape[1]))
    print("Number of rows: {}".format(data.shape[0]))

    return data, vars_old, vars_lost, vars_new


def prepare(data: pd.DataFrame, clear=False, dropna=False, onehot=False) -> pd.DataFrame:
    """
    Better not use it at all.
    This should be only part of the preprocessing pipeline; SciKit Learn best!
    Use to load data for modelling NOT for EDA, if necessary.

    clear: False; remove non-usable variables, i.e. those from VARS_TO_REMOVE ?
    dropna: False; drop all records with NaN? Dangerous !!!
    onehot: False; do apply one-hot encoding for factors?
    """
    #if all([not dropna, not onehot, not clear]):
    #    print("All `dropna`, `clear` and `onehot` are False so we do nothing.")

    if clear:
        data.drop(columns = su.VARS_TO_REMOVE, inplace=True, errors='ignore')
        print("\n+ Data not cleared -- unwanted variables removed.")
    else:
        print("\n! Data not cleared -- all variables left.")

    if dropna:
        data.dropna(inplace=True)
        print("\n+ Data cleared of NaN values -- only full records left.")
    else:
        print("\n! Data not cleared of NaN values -- all records left.")

    if onehot:
        data = pd.get_dummies(data, drop_first=True)
        print("\n+ One-hot encoding applied -- categorical variables changed to binary matrices.")
    else:
        print("\n! One-hot encoding not applied -- categorical variables left unprocessed.")

    return data


def raw_to_pickle(path_read: str = None,
        path_write: str = None,
        validate: bool = False,
        types: bool = False,
        clear: bool = False,
        dropna: bool = False,
        onehot: bool = False,
        ) -> None:
    """
    The same as .../app/new_data_prep.py/__main__ but to be used as script function.
    1. Loading raw data from .csv;
    2. Optionally:
        - validation
        - changing variables types according to COL_TYPES;
        - clearing of unnecessary variables according to VARS_TO_REMOVE;
        - dropping records with NaNs (danger!);
        - one-hot encoding for factors;
    4. witing to .pkl

    path_read: None; str; Path to .csv file to read data from. Relative to `.../data/`.
    path_write: None; Path to .pkl file to write prepared data. Relative to `.../data/`.
    """
    path_read = su.PATH_DATA / path_read if path_read else su.PATH_DATA_CSV
    data, vars_old, vars_lost, vars_new = read_raw(path_read)

    if all([not validate, not types, not clear, not dropna, not onehot]):
        print("All `validate`, `types`, `dropna`, `clear` and `onehot` are False so we do nothing\n" + \
              "but reading raw data from .csv and writing it to .pkl.")

    if validate:
        validation(data)
        print("\n+ Data validated.")
    else:
        print("\n! Data not validated.")

    if types:
        data = change_types(data)
        print("\n+ Data types changed.")
    else:
        print("\n! Data types left unchanged.")

    #if any([clear, dropna, onehot]):
    data = prepare(data, clear=clear, dropna=dropna, onehot=onehot)

    print()
    print("Data preprocessed to pickle")
    print("---------------------------")
    print("Number of columns: {}".format(data.shape[1]))
    print("Number of rows: {}".format(data.shape[0]))

    path_write = su.PATH_DATA / path_write if path_write else su.PATH_DATA_PKL
    data.to_pickle(path_write)

    print()
    print("Processed data written to {}".format(path_write))


#%%
def read(path: str = None) -> pd.DataFrame:
    """
    Reading to pd.DataFrame from file default or passed by the user.
    Default file is defined by dp.PATH_DATA_PKL : .../data/prep_data/data_prep.pkl.
    path : None; str
        path to .csv or .pkl data file relative to `.../data/` folder
        path="dane_2022-01-05.csv"  is possible or rather
        path="source_data/dane_2022-01-05.csv"
        If None then loads data from default .pkl.
    """
    if path is None:
        try:
            data = pd.read_pickle(su.PATH_DATA_PKL)
        except Exception as e:
            print("Default data file is not available and no other path to data was provided.")
            print(e)
            sys.exit(1)
    else:
        extension = str(path).split(".")[-1]
        try:
            if extension == "csv":
                data = read_raw(read_path = su.PATH_DATA / path)
            elif extension == "pkl":
                data = pd.read_pickle(su.PATH_DATA / path)
            else:
                Exception("Data file need to be '.csv' or '.pkl'.")
        except Exception as e:
            print("Provide path to data file is not available.")
            print(e)
            sys.exit(1)
    return data


def load_split(target="returnStd", variables=None, test_size=.3,
        path: str = None,
        clear=False, dropna=False, onehot=False) -> tuple:
    """
    As above -> SciKit Learn pipeline (or sth like this).
    path: None; str;
        path to data file to read from; relative to `.../data/`
        None means `.../prep_data/data_prep.pkl/`
    """
    xx = read(path)
    #print(xx.columns)
    if not variables is None:
        variables.append(target)
        xx = xx[variables]

    #if any([clear, dropna, onehot]):
    xx = prepare(xx, dropna, clear, onehot)

    yy = xx[target]
    xx.drop(target, inplace=True, axis=1)
    return train_test_split(xx, yy, test_size=test_size, random_state=1977)

#%%
