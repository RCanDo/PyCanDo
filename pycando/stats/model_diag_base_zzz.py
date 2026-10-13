#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Model Diagnostics class
version: 1.0
type: module
keywords: [model, metrics, diagnostics, validation, testing, reporting, ...]
description: |
    Container for model and all relative data
    constructed from Storage where paths/addresses to all objects are set.
remarks:
todo:
sources:
file:
    date: 2024-01-12
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import json
import yaml
from copy import deepcopy
from typing import Union
from pathlib import Path, PosixPath
import dill as pickle
import datetime as dt
import pypandoc

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from utils.stats import MetricsBin, MetricsReg
import utils.builtin as bi
from utils.project import to_dict
from common.storage import Storage


# %%
class ModelDiag(bi.Repr):
    """
    Class for the model diagnostics.
    Which model is determined by the only obligatory parameter, `storage`,
    where object of class Storage must be passed.
    This object contains all the information necessary to:
    - load the model object of interest,
    - load train and test data on which this model was trained and should be tested.
    Based on these objects the model is diagnosed where the result of it is a set of metrics
    stored in a dedicated object of class MetricsBin (for binary model) or MetricsReg (for regression model).
    Separate methods allow for generating reports (containing all metrics and some plots) in .md and .pdf format.
    These reports as well as .pkl with Metrics are stored in targets determined by `storage` passed at initialisation.

    Additionaly this class inherits from bi.Repr, i.e. is pretty printed.
    It's possible to accsess attributes via indexing (string names), and representing it all as dict (recursively).
    """

    def __init__(
            self,
            storage: Storage,
            calibrated: bool = False,
            stats: tuple[str] = None,
            threshold: float = None,
    ):
        """
        storage: Storage,
            object where all data sources and targets are defined; needs to be initialised saparately;
            paths to model object and all (train/test) data necessary for its diagnosis are stored in this object;
            however client is never forced to use them directly.
        calibrated: bool = False,
            if False then raw version of the model (internally called 'fit') is loaded and diagnosed.
            if True then calibrated version is loaded.
        stats: tuple[str] = None,
            collection of metrics (stats) to be calculated;
            only those which are predefined in respective Metrics class will be considered.
        threshold: float = .5,
            threshold for binary classifier (see MetricsBin help).
        """
        super().__init__()
        self.storage = storage

        self._stats = stats
        self.config = self.storage.load("config")

        self._init_threshold(threshold)

        self.binary = self.storage.objective in ('rtb', 'ecom')  # !!! ad-hoc

        self._calibrated = bi.coalesce(calibrated, storage.calibrated) if self.binary else False

        self.model = self.storage.load(self.stage)

        if self.binary:
            self.metrics = MetricsBin(self._stats, self._threshold)
        else:
            self.metrics = MetricsReg(self._stats)

        self.features, self.best_features = self._get_features()

        self.X_train, self.y_train = None, None
        self.X_valid, self.y_valid = None, None
        self.X_test, self.y_test = None, None
        self._get_data()

        self.y_train_hat, self.y_test_hat, self.y_valid_hat = self._get_predictions()

        self.performance()

    def _get_features(self) -> tuple[list, list | None]:
        """
        Returns
        features, best_features (if exist) on which the model was built.
        best_features must be extracted from model object which in turn may be nested in the pipeline object.

        """
        features = self.storage.load("xvars")

        if self.storage.exists("best_features"):
            best_features = self.storage.load("best_features")

        else:
            if isinstance(self.model, Pipeline):
                try:
                    best_features_idx = self.model['best_features_selection'].get_support(indices=True)
                    best_features = np.array(features)[best_features_idx].tolist()
                    self.storage.save(best_features, "best_features")
                except Exception as e:
                    print(e)
                    best_features = None
            else:
                best_features = None

        return features, best_features

    def _get_data(self) -> None:
        """
        Get all data on which model was built
        X_train, X_test, X_valid, y_train, y_test, y_valid
        It does not return, only fills already prepared (empty) entires.
        """
        self.__dict__.update(self.storage.load("split"))
        self.X_train = self.X_train[self.features]
        self.X_test = self.X_test[self.features]
        self.X_valid = self.X_valid[self.features] if len(self.X_valid) > 0 else self.X_valid

    def _get_importances(self, model):
        """
        Get vector of importances of all the features on which the model was built.
        If cannot find it in the model object then returns None.
        """
        try:
            importances = model.feature_importances_
        except Exception as e:
            print(e)
            importances = None
        return importances

    def _get_predictions(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Returns predictions from the model:
        y_train_hat, self.y_test_hat, self.y_valid_hat
        based on train, test and valid data (self.X_train, self.X_test, self.X_valid).
        """
        y_train_hat = self.predict(self.X_train)
        y_test_hat = self.predict(self.X_test)
        y_valid_hat = self.predict(self.X_valid) if len(self.X_valid) > 0 else pd.Series([])

        return y_train_hat, y_test_hat, y_valid_hat

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes prediction from self.model on predictors X.
        This prediction is made properly wrt type of the model (binary or regression).
        """
        if self.binary:
            y_hat = self.model.predict_proba(X)[:, 1]
        else:
            y_hat = self.model.predict(X)
        return pd.Series(y_hat)

    def performance(self) -> None:
        """
        Computes all the metrics for sel.model based on y_train, y_train_hat, y_test, y_test_hat.
        """
        self.metrics.compute(self.y_train, self.y_train_hat, self.y_test, self.y_test_hat)

    @property
    def stage(self):
        """ like 'step' of the modelling pipe but only two values possible: "fit" / "calibrated"
        """
        stage = "calibrated" if self._calibrated else "fit"
        return stage

    @property
    def file_stem(self):
        stem = f"{self.storage.objective}_{self.stage}"
        return stem

    @property
    def calibrated(self):
        return self._calibrated

    @calibrated.setter
    def calibrated(self, new_calibrated: bool):
        new_calibrated = new_calibrated if self.binary else False
        if self._calibrated != new_calibrated:
            print("Changing `calibrated` value. Model needs to be reloaded.")
            self._calibrated = new_calibrated
            self.model = self.storage.load(self.stage)
            self._get_predictions()
        else:
            print("The same `calibrated` as already set. No need for model reload.")

    def _init_threshold(self, threshold: float) -> None:
        self._threshold = bi.coalesce(threshold, self.config.get('threshold'), 0.1)
        self.config['threshold'] = self._threshold
        self.storage.save(self.config, "config")

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_thresh):
        if self.binary:
            if self._threshold != new_thresh:
                print("`threshold` changed -- all metrics needs to be recalculated ...")
                self._threshold = new_thresh
                self.metrics = MetricsBin(self._stats, self._threshold)
                self.performance()
        else:
            print("In case of regression model `threshold` is irrelevant.")

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, new_stats):
        if self._stats != new_stats:
            print("`stats` changed -- Metrics object needs to be recalculated ...")
            self._stats = new_stats
            if self.binary:
                self.metrics = MetricsBin(self._stats, self._threshold)
            else:
                self.metrics = MetricsReg(self._stats)
            self.performance()

    @property
    def importances(self):
        """
        We assume that:
        1. model is one of basic sklearn estimators, such that it has .feature_importances_ attribute
        2. or is CalibratedClassifierCV built on a sklearn.Pipeline with final step 'model' which is like 1.
        """
        if isinstance(self.model, CalibratedClassifierCV):
            model = self.model.estimator
        else:
            model = self.model

        if isinstance(model, Pipeline):
            importances = self._get_importances(model['model'])
        else:
            importances = self._get_importances(model)

        if importances is not None:
            importances = pd.Series(importances)
            if self.best_features:
                importances.index = self.best_features

            importances = importances.sort_values(ascending=False)

        return importances

    # %% repr
    def to_dict(self, df=False) -> dict:
        """"""
        if not df:
            dic = {k: to_dict(v) for k, v in self.__dict__.items() if not isinstance(v, pd.DataFrame)}
        else:
            dic = {k: to_dict(v) for k, v in self.__dict__.items()}
        return dic

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def copy(self):
        return deepcopy(self)

    # %% plots
    def plot_regression(
            self,
            y: pd.Series = None,
            y_hat: pd.Series = None,
            info: str = None,
            figsize: tuple[int] = (6, 6),
            path: str | Path = ".",
    ) -> Path:
        """
        Plots predicted values vs true values for regression model.

        Arguments
        ---------
        y: pd.Series = None,
            true value of a target
        y_hat: pd.Series = None,
            predicted value of a target.
        info: str = None,
            additional info to be printed within main figure title, just after "Regression".
        figsize: tuple[int] = (6, 6)
            size of a figure in inches, (width, height).
        path: str | Path = "."
            directory where the file with plot will be written

        Returns
        -------
        file: str
            name of a file to which the plot is stored (in a current working directory);
            this name is set automatically based on data from self.storage.
        """
        if info is None and y is None:
            info = "test"
        y = bi.coalesce(y, self.y_test)
        y_hat = bi.coalesce(y_hat, self.y_test_hat)
        plt.figure(figsize=figsize)
        #
        p1 = max(max(y), max(y_hat))
        p2 = min(min(y), min(y_hat))
        plt.scatter(y, y_hat, c='crimson', alpha=.2, s=4)
        plt.axline((p1, p1), (p2, p2), color='gray', linewidth=.7)
        plt.title(f"Regression {info}")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        #
        plt.tight_layout()
        file = f"{self.file_stem}-{info}.png" if info else f"{self.file_stem}.png"
        file = Path(path) / file
        plt.savefig(file)
        return file

    def plot_roc(
            self,
            y: pd.Series = None,
            y_hat: pd.Series = None,
            info: str = None,
            figsize: tuple[int] = (4, 4),
            path: str | Path = ".",
    ) -> Path:
        """
        ROC curve for binary model

        Arguments
        ---------
        y: pd.Series = None,
            true value of a target
        y_hat: pd.Series = None,
            predicted value of a target; this must be raw prediction (.predict_proba), not binarised yet.
        info: str = None,
            additional info to be printed within main figure title, just after "ROC".
        figsize: tuple[int] = (4, 4)
            size of a figure in inches, (width, height).
        path: str | Path = "."
            directory where the file with plot will be written

        Returns
        -------
        file: str
            name of a file to which the plot is stored (in a current working directory);
            this name is set automatically based on data from self.storage.
        """
        if info is None and y is None:
            info = "test"
        y = bi.coalesce(y, self.y_test)
        y_hat = bi.coalesce(y_hat, self.y_test_hat)
        plt.figure(figsize=figsize)
        #
        fpr, tpr, _ = roc_curve(y, y_hat)
        plt.plot(fpr, tpr)
        plt.axline((0, 0), (1, 1), color='gray', linewidth=.7)
        plt.title(f"ROC {info}")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        #
        plt.tight_layout()
        file = f"{self.file_stem}-{info}.png" if info else f"{self.file_stem}.png"
        file = Path(path) / file
        plt.savefig(file)
        return file

    def plot(
            self,
            data_type: str,
            path: str | Path = ".",
    ) -> str | Path:
        """
        Renders and stores diagnostics plots for train or test data,
        proper to the type of self.model: 'ROC' for binary and 'Regression' for regression model.

        Arguments
        ---------
        data_type: str
            one of 'train' or 'test'.
        path: str | Path = "."
            directory where the file with plot will be written

        Returns
        -------
        file: str
            name of a file to which the plot is stored (in a current working directory);
            this name is set automatically based on data from self.storage.
        """
        if data_type == "train":
            y = self.y_train
            y_hat = self.y_train_hat
        elif data_type == "test":
            y = self.y_test
            y_hat = self.y_test_hat
        else:
            raise ValueError(f'data_type can be "train" or "test", given {data_type}')
        if self.binary:
            file = self.plot_roc(y, y_hat, data_type, path=path)
        else:
            file = self.plot_regression(y, y_hat, data_type, path=path)
        return file

    # %% generating reports
    def report_md(
            self,
            file: str | PosixPath = None,
            precision: int = 3,
            path: str | Path = ".",
    ) -> tuple[Path, Path, Path]:
        """
        Generates report in MarkDown format (.md) on diagnosis of self.model.
        It contains all metrics (stats) stored in self.metrics (calculated at initialisation)
        as well as proper diagnostic plots for both test and train data.

        Renders and stores diagnostics plots for train or test data,
        proper to the type of self.model: 'ROC' for binary and 'Regression' for regression model.

        Arguments
        ---------
        file: str
            name of file to which the report shall be stored;
            if None then set automatically based on data from self.storage.
        precision: int = 3;
            number of digits to round results to;
        path: str | Path = "."
            directory where the file with report will be written

        Returns
        -------
        file_out: str,
            name of a file to which the report is stored (in a current working directory).
        fig_train: str,
            name of a file to which the plot for train data is stored (in a current working directory);
        fig_test: str
            name of a file to which the plot for test data is stored (in a current working directory);
       """

        fig_train = self.plot("train", path)
        fig_test = self.plot("test", path)

        stats_df = self.metrics.to_frame().round(precision)

        binary_stats = []
        if self.binary and isinstance(self.metrics, MetricsBin):
            confusion = self.metrics.confusion_df()
            confusion = confusion.reset_index().rename(columns={'level_0': ''})
            confusion = f"__Confusion matrix__, threshold = {self.threshold} \n\n" \
                        + confusion.to_markdown(index=False)

            if "nobs" in self.metrics._stats:
                df_target = pd.DataFrame(
                    [[0, self.metrics.nobs0["train"], self.metrics.nobs0["test"]],
                     [1, self.metrics.nobs1["train"], self.metrics.nobs1["test"]]],
                    columns=["target", "train", "test"])
                target = f"{df_target.to_markdown(index=False)}"
                target_title = "### Number of 0/1 in target for train and test sets"
                stats_df.drop(index="nobs0", axis=1, inplace=True)
                stats_df.drop(index="nobs1", axis=1, inplace=True)
            else:
                target, target_title = '', ''

            binary_stats = [confusion, target_title, target]

        features_importances = f"### Feature Importances\n, {self.importances.round(precision).to_markdown()}"\
                                if self.importances is not None else ""

        content = [
            f"# {self.storage.model_id} -- {self.storage.objective.upper()} -- {self.stage}",
            f"## Report {dt.datetime.now().isoformat()[:19]}",  # with/without Time Zone ?
            "### Statistics",
            stats_df.to_markdown(),
            *binary_stats,
            f"![train]({fig_train})",
            f"![test]({fig_test})",
            f"{features_importances}",
            #
            "### Config",
            "```\n" + yaml.dump(self.config, sort_keys=False, indent=4) + "\n```"
        ]

        file = file if file else self.file_stem
        file = Path(path) / file
        file_out = file.with_suffix(".md")

        with open(file, "wt") as f:
            f.write("\n\n".join(content))

        return file_out, fig_train, fig_test

    def report_pdf(
            self,
            file: str | PosixPath = None,
            precision: int = 3,
            path: str | Path = ".",
    ) -> Path:
        """
        Generates report in PDF format (.pdf) on diagnosis of self.model.
        It contains all metrics (stats) stored in self.metrics (calculated at initialisation)
        as well as proper diagnostic plots for both test and train data.

        PDF report is based on MarkDown report generated earlier, stored in a file `file` (extension neglected).
        If this is not the case
        (i.e. there is no file with name `file` and extension '.md' in current working directory)
        then self.report_md() is run to generate .md report.

        Arguments
        ---------
        file: str
            name of a file with MarkDown report from which .pdf will be generated,
            and to which the report shall be stored (with extension set to '.pdf');
            if None then set automatically based on data from self.storage.
        precision: int = 3;
            number of digits to round results to;
        path: str | Path = "."
            directory where the file with report will be written

        Returns
        -------
        file_pdf: Path,
            name of a file to which the report is stored (in a current working directory).
        """
        file = file if file else self.file_stem
        file = Path(path) / file
        file_md = file.with_suffix(".md")
        if not file_md.exists():
            self.report_md(file_md, precision, path)
        file_pdf = file.with_suffix(".pdf")
        pypandoc.convert_file(file_md, "pdf", outputfile=str(file_pdf), extra_args=['-V', 'geometry:margin=1.5cm'])
        return file_pdf

    def save_metrics(
            self,
            file: Union[str, PosixPath] = None,
            path: str | Path = ".",
    ) -> Path:
        """
        Arguments
        ---------
        file: str
            name of file to which the object with all metrics shall be stored;
            if None then set automatically based on data from `self.storage`.
            The object is `self.metrics` (which is of Metrics class) and is stored as .pkl
        path: str | Path = "."
            directory where the file with metrics will be written

        Returns
        -------
        file_out: Path,
            name of a file to which the metrics is stored (in a current working directory).
        """
        file = file if file else f"{self.file_stem}-metrics"
        file = Path(path) / file
        file_out = file.with_suffix(".pkl")
        pickle.dump(self.metrics, open(file_out, 'bw'))
        return file_out

    def save_feature_importances(
            self,
            file: Union[str, PosixPath] = None,
            path: str | Path = ".",
    ) -> Path:
        """
        Arguments
        ---------
        file: str
            name of file to which feature importances shall be stored;
            if None then set automatically based on data from `self.storage`.
        path: str | Path = "."
            directory where the file with metrics will be written

        Returns
        -------
        file_out: Path,
            name of a file to which the feature importances is stored (in a current working directory).
        """
        file = file if file else f"{self.file_stem}-feature_importances"
        file = Path(path) / file
        file_out = file.with_suffix(".json")
        importances_json = self.importances.to_json()
        with open(file_out, 'w') as f:
            json.dump(importances_json, f)
        return file_out

    # %%
    def report(
            self,
            name: str = None,
            storage: Union[str, bool] = 'move',
            precision: int = 3,
            path: str | Path = ".",
    ) -> None:
        """
        name: str = None
            name of the file under which report files will be stored (in .md and .pdf format);
            this name should not contain file extension (as each report file is in different format);
            however if extension is passed then it's ignored;
            if None (default) then the files name is taken from `self.storage` settings;
            Notice that plot files are stored under different names set automatically.
        storage: Union[str, bool] = 'move'
            if 'move' then moves all files to targets defined by 'self.storage';
            if 'copy' then copies "; i.e. all files are also left locally;
            if False (or None) then files are stored only in cwd (not moved or copied to targets from 'self.storage').
            True == 'copy'.
        precision: int = 3;
            number of digits to round results to;
        path: str | Path = "."
            directory where all the resulting files will be written;
            however, this is meant to be kind of _temporary_ directory for writing down results of each report step,
            ! as all the report files will be copied or moved (see `storage`) to the target directory and files
            defined within the `self.storage` object;
            e.g. `path` may be a temporary directory created by `TemporaryDirectory()` of `tempfile` library
            if we really don't want to retain results locally (value of `storage` param is then irrelevant).
        """

        file_md, fig_train, fig_test = self.report_md(name, precision, path=path)
        file_pdf = self.report_pdf(path=path)
        file_metrics = self.save_metrics(path=path)
        if self.importances is not None:
            file_importances = self.save_feature_importances(path=path)

        if storage:
            match storage:
                case True | 'copy':
                    fun = self.storage.copy
                case 'move':
                    fun = self.storage.move
                case _:
                    raise ValueError(f'Invalid storage file movement option; given `{storage}`.')

            fun(file_md, "report")
            fun(fig_train, "report")
            fun(fig_test, "report")
            fun(file_pdf, "report")
            fun(file_metrics, "report")
            if self.importances is not None:
                fun(file_importances, "report")
