#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: project utilities
version: 1.0
type: module
keywords: [project, parameters, documents, model, metrics, ...]
description: |
    Project management helpers;
    mainly data structures for storing all kind of parameters.
content:
    -
remarks:
todo:
    - TrainTestMetric, how to deal with non-scalar values of tests results ??? (~150)
    - ModelSpec is not updated on setting new id or name (~327)
    - ModelSpec.write() doesn't work now (~369)
    - ModelSpec.to_df() what about .remarks (~423)
sources:
file:
    usage:
        interactive: False
        terminal: False
    date: 2022-01-19
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
# import os
import sys
import json
sys.path.insert(1, "../")
# import pathlib
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
from functools import partialmethod

import numpy as np
import pandas as pd

# from scipy.stats import entropy
from sklearn.metrics import r2_score, f1_score, accuracy_score  # roc_auc_score, roc_curve,
from sklearn.metrics import mean_squared_error, confusion_matrix, explained_variance_score
from sklearn.model_selection import cross_val_score

# from utils.builtin import flatten, coalesce
import common.builtin as bi

from .helpers import to_dict

from dataclasses import dataclass, field  # fields
from typing import Any, List, Dict, Set  # , Iterable
from copy import deepcopy


# %%
# to be used as partialmethod to extend pd.DataFrame
def ravel_binary_confusion(self: pd.DataFrame, to="dict"):
    """meant only for BINARY confusion matrix with order of levels (F,T) or (-,+) or (0,1)
    """
    # print(self.shape)
    conf = self.to_numpy().ravel()
    tn, fp, fn, tp = conf.tolist()
    if to:
        if isinstance(to, bool):
            to = "dict"
        #
        if to == "dict":
            conf = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        elif to == "list":
            conf = [tn, fp, fn, tp]
        elif to == "tuple":
            conf = (tn, fp, fn, tp)
        elif to in ["nparray", "array"]:
            pass
        elif to in ["series", "pandas"]:
            conf = pd.Series({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
    else:
        # print(Exception("`ravel_binary_confusion` argument `to`
        #       must be one of 'dict', 'list', 'tuple', 'nparray' or 'array'"))
        conf = self
    return conf


class BinaryConfusion(pd.DataFrame):
    """"""
    def __init__(self, yy, yy_hat, as_int=True, *args, **kwargs):

        if as_int:
            try:
                yy = yy.astype(int)
                yy_hat = yy_hat.astype(int)
            except Exception as e:
                print(e)

        conf = confusion_matrix(yy, yy_hat)
        super().__init__(conf, *args, **kwargs)
        #
        self.index = ['-', '+']
        self.columns = ['-', '+']
        self.rename_axis(index="real", columns="prediction")

    ravel = partialmethod(ravel_binary_confusion)

    def to_series(self):
        return self.ravel("series")


# %%
@dataclass
class TrainTestMetric:
    test: Any = None
    train: Any = None

    def __repr__(self):
        ind = " " * 4

        def if_multi_liner(v):
            s = v.__repr__()
            if "\n" in s:
                s = "\n" + s
            s = s.replace("\n", "\n" + ind + " " * 2)
            return s

        res = "\n" + "\n".join(ind + f"{k} : {if_multi_liner(v)}" for k, v in self.__dict__.items())
        return res

    def to_dict(self):
        def _to_dict(x):
            try:
                return x.to_dict()
            except Exception:
                return x
        dic = {k: _to_dict(v) for k, v in self.__dict__.items()}
        return dic

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def to_series(self):
        """??? how to deal with non-scalar values ???
        """
        ss = pd.Series([self.test, self.train], index=("test", "train"))
        return ss


@dataclass(eq=False)
class Metrics:
    # evs: Any = field(default_factory=lambda: TrainTestMetric(),
    #                  metadata={"desc": "Explained Variance Score"})
    # r2: Any = field(default_factory=lambda: TrainTestMetric(),
    #                 metadata={"desc": "adjusted R2 (coeff. of determination"})
    # rmse: Any = field(default_factory=lambda: TrainTestMetric(),
    #                   metadata={"desc": "Residual Mean Squared Error"})
    # # ADD MORE !!!
    # accuracy: Any = field(default_factory=lambda: TrainTestMetric(),
    #                       metadata={"desc": "Accuracy score: #(proper predictions) / #all"})
    # f1: Any = field(default_factory=lambda: TrainTestMetric(),
    #                 metadata={"desc": "Residual Mean Squared Error"})
    # confusion: Any = field(default_factory=lambda: TrainTestMetric(),
    #                        metadata={"desc": "Residual Mean Squared Error"})
    # ??? divide into regression/classification subtypes ???

    @property
    def is_empty(self):
        return len(self.__dict__.keys()) == 0

    def __repr__(self):
        ind = " " * 3
        if not self.is_empty:
            res = "\n" + "\n".join(ind + f"{k} : {v!r}" for k, v in self.__dict__.items() if v is not None)
        else:
            res = ind + "no metrics yet"
        return res

    def to_dict(self):
        def _to_dict(x):
            try:
                return x.to_dict()
            except Exception:
                return x
        dic = {k: _to_dict(v) for k, v in self.__dict__.items() if v is not None}
        return dic

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def to_series(self, what=None):
        """"""
        if not self.is_empty:
            if what is None:
                what = [k for k in self.__dict__.keys() if not k == "confusion"]
            #
            ss = pd.concat(
                {k: v.to_series() for k, v in self.__dict__.items() if (k in what and v is not None)},
                axis=0)
        else:
            ss = pd.Series()
        return ss

    def to_frame(self, what=None):
        """"""
        if not self.is_empty:
            df = self.to_series(what).unstack()
        else:
            df = pd.DataFrame()
        return df

    # %%

    def _metrics_type(self, type: str) -> str:
        if type in ['regression', 'numeric', 'reg', 'num', 'r', 'n']:
            type = 'r'
        elif type in ['classification', 'factor', 'categorical', 'class', 'fac', 'cat', 'c', 'f']:
            type = 'c'
        elif type in ['binary', 'bin', 'b']:
            type = 'b'
        else:
            raise Exception("`type` must be one of ['regression', 'numeric', 'reg', 'num', 'r', 'n'] or"
                            " ['classification', 'factor', 'categorical', 'class', 'fac', 'cat', 'c', 'f']"
                            " or ['binary', 'bin', 'b'].")
        return type

    def metrics(self, yy, yy_hat, type, ravel="", ):
        """
        ! inplace !
        binary : relevant for `regression` and 'classification'
            if one wants to check binary version
            with mapping given by this arg, e.g. `binary = lambda x: x > 0`
            or `binary = lambda x: x in ['a', 'b', 'e']`;
            if None binary version is not checked;
        type : 'regression', 'classification', ...
        """

        if type == 'r':
            evs = explained_variance_score(yy, yy_hat)
            r2 = r2_score(yy, yy_hat)
            rmse = np.sqrt(mean_squared_error(yy, yy_hat))
            return evs, r2, rmse

        elif type == 'c':
            accuracy = accuracy_score(yy, yy_hat)
            f1 = f1_score(yy, yy_hat)
            confusion = confusion_matrix(yy, yy_hat)  # ! create Confusion class analog for BinaryConfusion
            return accuracy, f1, confusion

        elif type == 'b':
            # subset of the categorical but nicer __repr__
            accuracy = accuracy_score(yy, yy_hat)
            f1 = f1_score(yy, yy_hat)
            confusion = BinaryConfusion(yy, yy_hat).ravel(ravel)
            return accuracy, f1, confusion

    def compute(
            self, yy_train, yy_train_hat, yy_test=None, yy_test_hat=None,
            type=None, ravel=""):
        """"""
        # ! deafult `type` but better do not rely on it !
        type = bi.coalesce(type, 'regression')
        type = self._metrics_type(type)

        if type == 'r':
            self.evs = TrainTestMetric()
            self.r2 = TrainTestMetric()
            self.rmse = TrainTestMetric()
        elif type in ['c', 'b']:
            self.accuracy = TrainTestMetric()
            self.f1 = TrainTestMetric()
            self.confusion = TrainTestMetric()

        if type == 'r':
            self.evs.train, self.r2.train, self.rmse.train = \
                self.metrics(yy_train, yy_train_hat, type=type)
            if yy_test is not None:
                self.evs.test, self.r2.test, self.rmse.test = \
                    self.metrics(yy_test, yy_test_hat, type=type)
        elif type in ['c', 'b']:
            self.accuracy.train, self.f1.train, self.confusion.train = \
                self.metrics(yy_train, yy_train_hat, type=type, ravel="")
            if yy_test is not None:
                self.accuracy.test, self.f1.test, self.confusion.test = \
                    self.metrics(yy_test, yy_test_hat, type=type, ravel="")


# %%
@dataclass
class ModelSpec:
    name: str       # short and descriptive/suggestive/informative; usually like the file name in which it was created
    id: str         # we suggest "01", "02", ... just in case some minor versions stems from one main procedure
    #               # (within one file)
    type: str       # ~ (linear/logistic/...) regression / (binary/ternary/...)  classification
    library: str
    function: str
    target: str
    #
    name_id: str = None
    #
    transform: bool = False   # indicate if su.TRANSFORMATIONS are in use
    remarks: List[str] = field(default_factory=lambda: [])
    #
    success: Any = None                # relevant only for binary target
    unwanted_target_values: Any = None
    #
    targets: Set[str] = field(default_factory=lambda: set())
    ids: Set[str] = field(default_factory=lambda: set())
    predictors_0: Set[str] = field(default_factory=lambda: set())  # before one-hot and other transformations
    predictors: Set[str] = field(default_factory=lambda: set())    # final -- ready to modelling
    to_remove_0: Set[str] = field(default_factory=lambda: set())   # will be immediately removed from `predictors_0`
    to_remove: Set[str] = field(default_factory=lambda: set())
    #
    exclude: Dict = field(default_factory=lambda: dict())
    #   # {"Variable": [..values..]}  -- remove records where Variable takes any of [..values..]
    #
    nulls_threshold: float = .8
    most_common_threshold: float = .8
    data_fraction: float = 1.
    test_size: float = .3
    #
    _root: str = Path(".")   # !!!  ROOT folder of the PROJECT -- always expected to be working directory !!!
    #                   # -> should be recorded as  absolute  path in .../common/setup/config.py: ROOT
    # all paths below are relative (wrt. current dir) except 'data' which is relative to `_root` provided BUT...
    folder: str = None  # folder for all the model outputs;
    #                   # best if == self.name == name of the file in which model is constructed
    # ??? data_version: str = None
    path: Dict[str, str] = field(
        default_factory=lambda: dict(
            data_raw=Path("data/data.pkl"),
            # !!! it's NOT the same as [root]/data/.../raw/ (which is for raw data from client)
            # it is for data ready for input into model file (like xgboost_01.py)
            # but still need some prep. like OH, substitutions, removing NAa, splitting into test/train
            # (but no aggregations!);
            # !!! starting from  self._root !!! i.e. project root dir
            #
            data=None,
            model=None,
            info=None,
            transformations=None))
    #
    data: Dict = field(
        default_factory=lambda: dict(
            info_0=None,
            summary_0=None,     # last stage before transformations
            info=None,
            summary=None))      # last stage before modelling

    #
    model: Dict = field(
        default_factory=lambda: dict(
            call=None,
            summary=None))
    #
    randstate: int = 14521519

    def __post_init__(self):
        """"""
        self.update()

    def update(self):
        """
        !!! it is not updated on setting new id or name !!!
        setters/getters don't fit neatly with dataclass :( although it is possible
        """
        #
        self.name_id = f'{self.name}_{self.id}'
        self.predictors_0 = self.predictors_0.difference(self.to_remove_0)
        #
        # PATHS -------------
        self.path['data_raw'] = self._root / self.path['data_raw']
        #
        if self.folder is None:
            self.folder = self._root / "models" / self.name
        else:
            self.folder = Path(self.folder)
        self.folder.mkdir(exist_ok=True, mode=511)  # default mode
        #
        self.path['model'] = self.folder / (self.name_id + '.pkl')
        self.path['cv_grid'] = self.folder / (self.name_id + '_cv_grid.pkl')
        #
        self.path['info'] = self.folder / (self.name_id + '.json')
        #
        # ! data and transformations should be the same for all id's
        # BUT it's always possible to change that
        self.path['data'] = self.folder / (self.name + '_data.pkl')
        self.path['transformations'] = self.folder / (self.name + '_transformations.dill')

    def to_dict(self, df=False):
        self.update()

        if not df:
            dic = {k: to_dict(v) for k, v in self.__dict__.items() if not isinstance(v, pd.DataFrame)}
        else:
            dic = {k: to_dict(v) for k, v in self.__dict__.items()}
        return dic

    def write(self, path=None):
        """!!! doesn't work now !!!
        writing itself to .json
        """
        self.update()
        path = self.path['info'] if path is None else path
        with open(path, 'tw') as f:
            json.dump(self.to_dict(df=True), f)

    def __getitem__(self, name):
        return self.__getattribute__(name)

    # %% representation
    def __str__(self) -> str:
        return bi.indent(self, head=6)

    def __repr__(self) -> str:
        return bi.indent(self, head=6)

    def print(self, *args, **kwargs) -> None:
        print(bi.indent(self, *args, **kwargs))

    # %%
    def copy(self):
        self.update()
        return deepcopy(self)

    def to_df(self):
        """from this one may get a lot of formats
        ??? what about .remarks -- a list!
        """
        self.update()
        return pd.DataFrame(self.__dict__)

    def add_remark(self, txt):
        """"""
        self.remarks.append(txt)

    def set_remark(self, n, txt):
        """
        n : int > 0
            id of remark = position on the list - but numbering from 1 !
        """
        self.remarks[n - 1] = txt

    def print_remarks(self):
        if len(self.remarks) > 0:
            for i, r in enumerate(self.remarks):
                print(f" remark {i+1}:")
                print(r)

    # %%
    @staticmethod
    def _apply_transform(trans, *args):
        def trans_or_none(y):
            z = None if y is None else trans(y)
            return z
        res = [trans_or_none(y) for y in args]
        return res

    def _metrics_type(self, type: str) -> str:
        if type in ['regression', 'numeric', 'reg', 'num', 'r', 'n']:
            type = 'r'
        elif type in ['classification', 'factor', 'categorical', 'class', 'fac', 'cat', 'c', 'f']:
            type = 'c'
        elif type in ['binary', 'bin', 'b']:
            type = 'b'
        else:
            raise Exception("`type` must be one of ['regression', 'numeric', 'reg', 'num', 'r', 'n'] or"
                            " ['classification', 'factor', 'categorical', 'class', 'fac', 'cat', 'c', 'f']"
                            " or ['binary', 'bin', 'b'].")
        return type

    def performance(
            self, yy_train, yy_train_hat, yy_test=None, yy_test_hat=None,
            type=None,   # bool, variables as categories or numeric; if None uses yy_train.dtype to figure out
            binary=True,
            inverse_trans=None,
            ravel=False):
        """
        type: str, None
            what type of metrics to compute, what is equivalent to
            what type of model do we deal with;
            possible values are:
            - one of 'classification', 'factor', 'categorical', 'class', 'fac', 'cat'
              for classification model i.e. categorical/factor target/response;
              then metrics like  f1, accuracy and confusion matrix  will be calculated;
            - one of 'regression', 'numeric', 'reg', 'num'
              for `regression` model i.e. numeric target/response;
              then metrics like  evs (Explained Variance Score), r2, rmse  will be calculated;
            - 'binary' is for binary classification;
              the same metrics as for ordinary classififactiona are calculated
              but confusion matrix is better formatted;
            - None (default) - then the type of metrics (model) is inferred from
              type of `yy_train`;
        binary: None, bool or binary function;
            calculate binary confusion matrix for binary version of data (and model)?
            this is irrelevevant if `type` is 'binary';
            otherwise, if True (default) or binary function,
            calculates binary model metrics, additionally wrt. main model type;
            if `type='regression'` and `binary=True` then binary model
            is calculated from `yy_` transformed by f(y)=(y>0) or
                binary = lambda y: y>0;
            if `type='classification'` and `binary=True` then binary model
            is calculated from `yy_` transformed by f(y)=(y==c0)
            where c0 is most common value for `yy_train`, i.e.
                binary = lambda y: y==c0;
            # in this case (`type='regression'`)
            obviously one may pass to `binary` any binary function in need, like e.g.
                binary = lambda y: y in ['c', 'g', 'z'];
            if None or False binary version is not calculated;
        inverse_trans: callable (function)
            if model is calculated on transformed data then one may pass
            inverse transformation to check the performance for target (`yy_train`, `yy_test`)
            and predictions (`yy_tain_hat`, `yy_test_hat`)
            transformed back to original values;
            such performance is usually much worse then for transformed data
            (on which model was built).
        """

        if type is None:
            if (yy_train.dtype in ['category', 'string', 'object']):
                type = 'c'
            elif (yy_train.nunique() == 2):
                type = 'b'
            else:
                type = 'r'

        type = self._metrics_type(type)

        binary = bi.coalesce(binary, False)
        if type == 'b':
            binary = False

        # some defaults for `binary` if it's True
        if binary and not callable(binary):
            if type == 'r':
                binary = lambda x: x > 0
            if type == 'c':
                cat0 = yy_train.value_counts(sort=True).index[0]
                binary = lambda x: x == cat0

        # now `binary` is False or callabe

        self.metrics = Metrics()
        self.metrics.compute(yy_train, yy_train_hat, yy_test, yy_test_hat, type, ravel)

        if binary:
            yy_bin_train, yy_bin_train_hat, yy_bin_test, yy_bin_test_hat = \
                self._apply_transform(inverse_trans, yy_train, yy_train_hat, yy_test, yy_test_hat)

            self.metrics_bin = Metrics()
            self.metrics_bin.compute(yy_bin_train, yy_bin_train_hat, yy_bin_test, yy_bin_test_hat, 'b', ravel)

        # metrics on original target i.e. after inverse transformation (if transformation was applied)
        if inverse_trans is not None:

            self.metrics_raw = Metrics()
            yy_raw_train, yy_raw_train_hat, yy_raw_test, yy_raw_test_hat = \
                self._apply_transform(inverse_trans, yy_train, yy_train_hat, yy_test, yy_test_hat)
            self.metrics_raw.compute(yy_raw_train, yy_raw_train_hat, yy_raw_test, yy_raw_test_hat, type, binary, ravel)

    def cv_score(self, model, xx, yy, scoring, cv=5):
        """scoring function (scorer) like in
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """
        if "cv" not in self.__dict__:
            self.cv = dict()

        if isinstance(scoring, list):
            for score in scoring:
                self.cv[score] = cross_val_score(model, xx, yy, cv=5, scoring=score)
        else:
            self.cv[scoring] = cross_val_score(model, xx, yy, cv=5, scoring=scoring)


# %%
