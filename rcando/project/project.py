# -*- coding: utf-8 -*-
#! python3
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

#%%
import os, sys, json, pickle
sys.path.insert(1, "../")
import pathlib
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
from functools import partialmethod

import numpy as np
import pandas as pd

#from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, confusion_matrix, explained_variance_score
from sklearn.model_selection import cross_val_score

#from utils.builtin import flatten, coalesce

from .helpers import to_dict

##
from dataclasses import dataclass, field, fields
from typing import Any, List, Dict, Set, Iterable
from copy import deepcopy

#%%
# to be used as partialmethod to extend pd.DataFrame
def ravel_binary_confusion(self: pd.DataFrame, to="dict"):
    """meant only for BINARY confusion matrix with order of levels (F,T) or (-,+) or (0,1)"""
    #print(self.shape)
    conf = self.to_numpy().ravel()
    tn, fp, fn, tp = conf.tolist()
    if to:
        if isinstance(to, bool):
            to = "dict"
        #
        if to=="dict":
            conf = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        elif to=="list":
            conf = [tn, fp, fn, tp]
        elif to=="tuple":
            conf = (tn, fp, fn, tp)
        elif to in ["nparray", "array"]:
            pass
        elif to in ["series", "pandas"]:
            conf = pd.Series({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
    else:
        #print(Exception("`ravel_binary_confusion` argument `to` must be one of 'dict', 'list', 'tuple', 'nparray' or 'array'"))
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
        self.index=['-', '+']
        self.columns=['-', '+']
        self.rename_axis(index="real", columns="prediction")

    ravel = partialmethod(ravel_binary_confusion)

    def to_series(self):
        return self.ravel("series")

#%%

@dataclass
class TrainTestMetric:
    test: Any = None
    train: Any = None

    def __repr__(self):
        ind = " "*4
        def if_multi_liner(v):
            s = v.__repr__()
            if "\n" in s:
                s = "\n" + s
            s = s.replace("\n", "\n" + ind + " "*2)
            return s
        res = "\n" + "\n".join(ind + f"{k} : {if_multi_liner(v)}" for k, v in self.__dict__.items())
        return res

    def to_dict(self):
        def to_dict(x):
            try:
                return x.to_dict()
            except:
                return x
        dic = {k: to_dict(v) for k, v in self.__dict__.items()}
        return dic

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def to_series(self):
        """TODO: how to deal with non-scalar values ???
        """
        ss = pd.Series([self.test, self.train], index=("test", "train"))
        return ss


@dataclass(eq=False)
class Metrics:
    evs: Any = field(default_factory = lambda: TrainTestMetric(), metadata={"desc": "Explained Variance Score"})
    r2: Any = field(default_factory = lambda: TrainTestMetric(), metadata={"desc": "adjusted R2 (coeff. of determination"})
    rmse: Any = field(default_factory = lambda: TrainTestMetric(), metadata={"desc": "Residual Mean Squared Error"})
    # TODO : ADD MORE !!!
    accuracy: Any = field(default_factory = lambda: TrainTestMetric(), metadata={"desc": "Accuracy score: #(proper predictions) / #all"})
    f1: Any = field(default_factory = lambda: TrainTestMetric(), metadata={"desc": "Residual Mean Squared Error"})
    confusion: Any = field(default_factory = lambda: TrainTestMetric(), metadata={"desc": "Residual Mean Squared Error"})
    #??? divide into regression/classification subtypes ???

    def __repr__(self):
        ind = " "*3
        res = "\n" + "\n".join(ind + f"{k} : {v!r}" for k, v in self.__dict__.items() if not v is None)
        return res

    def to_dict(self):
        def to_dict(x):
            try:
                return x.to_dict()
            except:
                return x
        dic = {k: to_dict(v) for k, v in self.__dict__.items() if not v is None}
        return dic

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def to_series(self, what=None):
        """"""
        if what is None:
            what = [k for k in self.__dict__.keys() if not k=="confusion"]
        #
        ss = pd.concat(
            {k: v.to_series() for k, v in self.__dict__.items() if (k in what and not v is None)},
            axis=0)
        return ss

    def to_frame(self, what=None):
        """"""
        df = self.to_series(what).unstack()
        return df


    def metrics(self, yy, yy_hat, binary=lambda x: x>0, ravel=""):
        """"""
        evs = explained_variance_score(yy, yy_hat)
        r2 = r2_score(yy, yy_hat)
        rmse = np.sqrt(mean_squared_error(yy, yy_hat))
        if binary:
            yy_bin = binary(yy)
            yy_bin_hat = binary(yy_hat)
            #
            accuracy = accuracy_score( yy_bin, yy_bin_hat )
            f1 = f1_score( yy_bin, yy_bin_hat )
            confusion = BinaryConfusion( yy_bin, yy_bin_hat ).ravel(ravel)
        else:
            accuracy, f1, confusion = None, None

        return evs, r2, rmse, accuracy, f1, confusion


    def compute(self,
            yy_train, yy_train_hat, yy_test=None, yy_test_hat=None,
            binary=lambda x: x>0, ravel=""):
        """"""
        self.evs.train, self.r2.train, self.rmse.train, self.accuracy.train, self.f1.train, self.confusion.train = \
            self.metrics(yy_train, yy_train_hat, binary=binary, ravel="")

        if not yy_test is None:
            self.evs.test, self.r2.test, self.rmse.test, self.accuracy.test, self.f1.test, self.confusion.test = \
                self.metrics(yy_test, yy_test_hat, binary=binary, ravel="")



@dataclass(eq=False)
class MetricsCat:
    f1: Any = field(default_factory = lambda: TrainTestMetric(None, None), metadata={"desc": "F1 score"})
    confusion: Any = field(default_factory = lambda: TrainTestMetric(None, None), metadata={"desc": "confision matrix"})
    # TODO : ADD MORE !!!
    #??? divide into regression/classification subtypes ???

    def __repr__(self):
        ind = " "*3
        res = "\n" + "\n".join(ind + f"{k} : {v!r}" for k, v in self.__dict__.items())
        return res

    def to_dict(self):
        def to_dict(x):
            try:
                return x.to_dict()
            except:
                return x
        dic = {k: to_dict(v) for k, v in self.__dict__.items()}
        return dic

    def __getitem__(self, name):
        return self.__getattribute__(name)


@dataclass
class ModelSpec:
    name : str      # short and descriptive/suggestive/informative; usually like the file name in which it was created
    id : str        # we suggest "01", "02", ... just in case some minor versions stems from one main procedure (within one file)
    type : str      # ~ (linear/logistic/...) regression / (binary/ternary/...)  classification
    library : str
    function : str
    target : str
    #
    name_id : str = None
    #
    transform: bool = False   # indicate if su.TRANSFORMATIONS are in use
    remarks : List[str] = field(default_factory=lambda: [])
    #
    success : Any = None                # relevant only for binary target
    unwanted_target_values : Any = None
    #
    targets : Set[str] = field(default_factory=lambda: set())
    ids : Set[str] = field(default_factory=lambda: set())
    predictors_0 : Set[str] = field(default_factory=lambda: set())  # before one-hot and other transformations
    predictors : Set[str] = field(default_factory=lambda: set())    # final -- ready to medelling
    to_remove_0 : Set[str] = field(default_factory=lambda: set())   # will be immediately removed from `predictors_0`
    to_remove : Set[str] = field(default_factory=lambda: set())
    #
    nulls_threshold : float = .8
    most_common_threshold : float = .8
    data_fraction : float = 1.
    test_size : float = .3
    #
    _root : str = None  #!!!  ROOT folder of the PROJECT  --  don't use it  !!!
                        # -> should be recorded as  absolute  path in .../utils/setup/config.py : ROOT
    # all paths below are relative (wrt. current dir) except 'data' which is relative to `_root` provided BUT...
    folder : str = None # folder for all the model outputs; best if == self.name == name of the file in which model is constructed
    path : Dict[str, str] = field(
        default_factory = lambda: dict(
            data_raw = Path('data/prep_data/data_prep.pkl'),
                        #!!! starting from  self._root !!!  see self.update()
                        # if self._root is None then full path to data should be provided
            data = None,
            model = None,
            info = None,
            transformations = None
            )
        )
    #
    data : Dict = field(
        default_factory = lambda: dict(
            info_0 = None,
            summary_0 = None,   # last stage before transformations
            info = None,
            summary = None      # last stage before modelling
            )
        )
    #
    model : Dict = field(
        default_factory = lambda: dict(
            call = None,
            summary = None,
            )
        )
    #
    randstate : int = 14521519


    def __post_init__(self):
        """"""
        self.update()


    def update(self):
        """
        TODO: it is not updated on setting new id or name !!!
        setters/getters doesn't fit neatly with dataclass :( although it is possible
        """
        #
        self.name_id = f'{self.name}_{self.id}'
        self.predictors_0 = self.predictors_0.difference(self.to_remove_0)
        #
        # PATHS -------------
        if self._root is None:
            self._root = Path("..")                      # assume we work from .../models/ so need to go up one level
            self.path['data_raw'] = self._root.joinpath(self.path['data_raw'])
        #
        if self.folder is None:
            self.folder = Path(".").joinpath(self.name)
        else:
            self.folder = Path(self.folder)
        #
        if not os.path.exists(self.folder):
            self.folder.mkdir()
        #
        self.path['model'] = self.folder.joinpath(self.name_id + '.pkl')
        self.path['cv_grid'] = self.folder.joinpath(self.name_id + '_cv_grid.pkl')
        #
        self.path['info'] = self.folder.joinpath(self.name_id + '.json')
        #
        #! data and transformations should be the same for all id's
        # BUT it's always possible to change that
        self.path['data'] = self.folder.joinpath(self.name + '_data.pkl')
        self.path['transformations'] = self.folder.joinpath(self.name + '_transformations.dill')


    def to_dict(self, df=False):
        self.update()

        if not df:
            dic = {k: to_dict(v) for k, v in self.__dict__.items() if not isinstance(v, pd.DataFrame)}
        else:
            dic = {k: to_dict(v) for k, v in self.__dict__.items()}
        return dic


    def write(self, path=None):
        """TODO : doesn't work now !!!
        writing itself to .json
        """
        self.update()
        path = self.path['info'] if path is None else path
        with open(path, 'tw') as f:
            json.dump(self.to_dict(df=True), f)


    def __getitem__(self, name):
        return self.__getattribute__(name)

    #%%

    def _not_too_long(self, obj, mx=33):
        try:
            ret = max(obj.shape) < mx
        except AttributeError:
            try:
                ret = len(obj) < mx
            except TypeError:
                ret = True
        return ret

    def __repr_attr__(self, attr, indent=1):
        """
        it is assumed that
        """
        if isinstance(attr, dict):
            res = "\n" + "\n".join(" "*3*indent + f"{k} : {self.__repr_attr__(v, indent+1)}" for k, v in attr.items())
        elif isinstance(attr, pd.DataFrame):
            res = "... pd.DataFrme ..."
        elif isinstance(attr, str):
            res = f"{attr!r}"
        elif isinstance(attr, Iterable):
            if self._not_too_long(attr):
                res = "\n" + " "*3*indent + f"{attr!r}"
            else:
                res = "... too long to display ..."
        else:     # what it could be ???
            if self._not_too_long(attr):
                res = f"{attr!r}"
            else:
                res = "... too long to display ..."
        return res

    def __repr__(self):
        self.update()
        res = "\n".join(f"{k} : {self.__repr_attr__(v)}" for k, v in self.__dict__.items())
        return res

    def copy(self):
        self.update()
        return deepcopy(self)

    def to_df(self):
        """from this one may get a lot of formats
        TODO:  what about .remarks -- a list!
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
        self.remarks[n-1] = txt

    def print_remarks(self):
        if len(self.remarks) > 0:
            for i, r in enumerate(self.remarks):
                print(f" remark {i+1}:")
                print(r)


    #%%
    def confusion_bin(self, yy, yy_hat, ravel=False, as_int=True):
        """works only for binary case"""

        if as_int:
            try:
                yy = yy.astype(int)
                yy_hat = yy_hat.astype(int)
            except Exception as e:
                print(e)

        conf = confusion_matrix(yy, yy_hat)
        conf = pd.DataFrame(conf, index=['-', '+'], columns=['-', '+']).rename_axis(index="real", columns="prediction")
        conf.ravel = partialmethod(ravel_binary_confusion)

        if ravel:
            if isinstance(ravel, str):
                conf = conf.ravel(to=ravel)
            else:
                conf = conf.ravel()

        return conf


    def metrics_reg(self, yy_train, yy_train_hat, yy_test=None, yy_test_hat=None,
            inverse_trans=None,   # function -- inverse transformation to apply (if forward was applied)
            binary=lambda x: x>0,  # calculate binary confusion matrix ?  turns continuous data to binary by  f(x) = (x>0)
            ravel="",
            #ret=False
            ):
        """"""
        # on train set

        self.metrics = Metrics()
        self.metrics.compute(yy_train, yy_train_hat, yy_test, yy_test_hat, binary, ravel)

        # metrics on original target i.e. after inverse transformation (if transformation was applied)
        if not inverse_trans is None:

            self.metrics_raw = Metrics()

            yy_raw_train = inverse_trans(yy_train)
            yy_raw_train_hat = inverse_trans(yy_train_hat)

            if not yy_test is None:
                yy_raw_test = inverse_trans(yy_test)
                yy_raw_test_hat = inverse_trans(yy_test_hat)
            else:
                yy_raw_test, yy_raw_test_hat = None, None

            self.metrics_raw.compute(yy_raw_train, yy_raw_train_hat, yy_raw_test, yy_raw_test_hat, binary, ravel)


    def metrics_cat(self, yy_train, yy_train_hat, yy_test=None, yy_test_hat=None, ravel=False):

        self.metrics = MetricsCat()

        self.metrics.f1.train = f1_score(yy_train, yy_train_hat)
        conf = confusion_matrix(yy_train, yy_train_hat)
        self.metrics.confusion.train = conf.ravel if ravel else conf

        if not yy_test is None:
            conf = confusion_matrix(yy_test, yy_test_hat)
            self.metrics.f1.test = f1_score(yy_test, yy_test_hat)
            self.metrics.confusion.test = conf.ravel if ravel else conf


    def performance(self, yy_train, yy_train_hat, yy_test=None, yy_test_hat=None,
            inverse_trans=None,
            binary=lambda x: x>0,  # calculate binary confusion matrix ?  turns continuous data to binary by  f(x) = (x>0)
            cat=None,   # variables as categories or numeric; if None uses yy_train.dtype to figure out
            ravel=False
            ):
        """"""
        # on train set

        if cat is None:
            cat = yy_train.dtype == 'category'

        if cat:

            self.metrics_cat(    ## not implemented yet!
                yy_train, yy_train_hat,
                yy_test, yy_test_hat,
                ravel
                )

        else:

            self.metrics_reg(
                yy_train, yy_train_hat,
                yy_test, yy_test_hat,
                inverse_trans,
                binary,  # calculate binary confusion matrix ?  turns continuous data to binary by  f(x) = (x>0)
                ravel
                )


    def cv_score(self, model, xx, yy, scoring, cv=5):
        """scoring function (scorer) like in  https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """
        if not "cv" in self.__dict__:
            self.cv = dict()

        if isinstance(scoring, list):
            for score in scoring:
                self.cv[score] = cross_val_score(model, xx, yy, cv=5, scoring=score)
        else:
            self.cv[scoring] = cross_val_score(model, xx, yy, cv=5, scoring=scoring)


#%%
