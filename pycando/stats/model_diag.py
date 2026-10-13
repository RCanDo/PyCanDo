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
    date: 2022-06-01
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scikitplot.metrics import plot_cumulative_gain, plot_lift_curve
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from utils.stats import MetricsBin, MetricsReg
import utils.builtin as bi
from utils.project import to_dict
from utils.plots import plot_covariates
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
            calibrated: bool = None,
            stats: tuple[str] = None,
            threshold: float = None,
            q_order: int = None,
            threshold_separately: bool = None
    ):
        """
        storage: Storage,
            object where all data sources and targets are defined; needs to be initialised saparately;
            paths to model object and all (train/test) data necessary for its diagnosis are stored in this object;
            however client is never forced to use them directly.
        calibrated: bool = None,
            if False then raw version of the model (internally called 'fit') is loaded and diagnosed.
            if True then calibrated version is loaded.
        stats: tuple[str] = None,
            collection of metrics (stats) to be calculated;
            only those which are predefined in respective Metrics class will be considered.
        threshold: float = None,
            threshold for binary classifier (see MetricsBin help).
        q_order: int = None,
            order of quantiles for some statistics, like lift and gain.
        """
        super().__init__()
        self.storage = storage

        self._stats = stats
        self.config = self.storage.load("config")
        self._init_threshold(threshold)
        self._init_q_order(q_order)

        # self.separately = False if threshold is not None or threshold_separately is None else threshold_separately
        self.separately = threshold_separately

        self.binary = self.storage.objective != 'val'  # !!! ad-hoc

        self._calibrated = bi.coalesce(calibrated, storage.calibrated, True) if self.binary else False

        self.model = self.storage.load(self.stage)

        if self.binary:
            self.metrics = MetricsBin(self._stats, self._threshold)
        else:
            self.metrics = MetricsReg(self._stats)

        self.features, self.best_features = self._get_features()

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.X_valid, self.y_valid = None, None
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
                    all_features_after_transform = \
                        [name.split("__")[1] for name in self.model['transform'].get_feature_names_out()]
                    best_features = np.array(all_features_after_transform)[best_features_idx].tolist()
                    # best_features = np.array(features)[best_features_idx].tolist()
                    self.storage.save(best_features, "best_features")
                except Exception as e:
                    print(e)
                    best_features = None
            elif isinstance(self.model, XGBClassifier) or isinstance(self.model, CalibratedClassifierCV):
                try:
                    best_features = features
                except Exception as e:
                    print(e)
                    best_features = None
            else:
                best_features = None

        return features, best_features

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
        return pd.Series(y_hat, index=X.index)

    def _yyhat(
            self, mode: str, numpy: bool = False
    ) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
        """
        mode: str,
            possible values: "train" / "test"
        numpy: bool = False
            if False, y and y_hat are returned as pd.Series;
            if True, y and y_hat are returned as np.arrays,
            and y_hat is 2-dim array: `np.array([1 - y_hat, y_hat]).T`
            as original prediction from binary model, giving scores for 0 and 1 in each row;

        """
        if mode == "train":
            y = self.y_train
            y_hat = self.y_train_hat
        elif mode == "test":
            y = self.y_test
            y_hat = self.y_test_hat
        else:
            raise ValueError(f'mode can be "train" or "test", given {mode}')

        if self.binary and numpy:
            y = y.values
            y_hat = y_hat.values
            y_hat = np.array([1 - y_hat, y_hat]).T

        return y, y_hat

    def performance(self, separately: bool = False) -> None:
        """
        Computes all the metrics for sel.model based on y_train, y_train_hat, y_test, y_test_hat.
        """
        if separately:
            self.metrics._compute_0("test", self.y_test, self.y_test_hat, self.metrics.threshold_opt["test"])
            self.metrics._compute_0("train", self.y_train, self.y_train_hat, self.metrics.threshold_opt["train"])
        else:
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
                print("`threshold` changed -- some metrics needs to be recalculated ...")
                self._threshold = new_thresh
                self.metrics = MetricsBin(self._stats, self._threshold, self._q_order)
                self.performance()
                # print(self._threshold)
        else:
            print("In case of regression model `threshold` is irrelevant.")

    def _init_q_order(self, q_order: float) -> None:
        self._q_order = bi.coalesce(q_order, self.config.get('q_order'), 10)
        self.config['q_order'] = self._q_order
        self.storage.save(self.config, "config")

    @property
    def q_order(self):
        return self._q_order

    @q_order.setter
    def q_order(self, new_q_order):
        if self.binary:
            if self._q_order != new_q_order:
                print("`q_order` changed -- some metrics needs to be recalculated ...")
                self._q_order = new_q_order
                self.metrics = MetricsBin(self._stats, self._threshold, self._q_order)
                self.performance()
        else:
            print("In case of regression model `q_order` is irrelevant.")

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, new_stats):
        if self._stats != new_stats:
            print("`stats` changed -- Metrics object needs to be recalculated ...")
            self._stats = new_stats
            if self.binary:
                self.metrics = MetricsBin(self._stats, self._threshold, self._q_order)
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

    def __save_fig(self, mode: str, path: str | Path = ".") -> Path | str:
        plt.tight_layout()
        if path:
            file = f"{self.file_stem}-{mode}.png" if mode else f"{self.file_stem}.png"
            file = Path(path) / file
            plt.savefig(file)
        else:
            file = ""
        return file

    def plot_regression(
            self,
            mode: str = None,
            path: str | Path = ".",
            figsize: tuple[int] = (6, 6),
    ) -> Path | str:
        """
        Plots predicted values vs true values for regression model.

        Arguments
        ---------
        mode: str
            one of 'train' or 'test'.
        path: str | Path = "."
            directory where the file with plot will be written;
            to not write the plot to file
            empty string "" may be passed (or anything which evaluates to `False` in `if` statement);
        figsize: tuple[int] = (6, 6)
            size of a figure in inches, (width, height).

        Returns
        -------sh
        file: Path | str
            name of a file to which the plot is stored (in a current working directory);
            this name is set automatically based on data from self.storage;
            empty string (meaning the plot was not written to file) if `path=""`.
        """
        y, y_hat = self._yyhat(mode)
        plt.figure(figsize=figsize)
        #
        p1 = max(max(y), max(y_hat))
        p2 = min(min(y), min(y_hat))
        plt.scatter(y, y_hat, c='crimson', alpha=.2, s=4)
        plt.axline((p1, p1), (p2, p2), color='gray', linewidth=.7)
        plt.title(f"Regression {mode}")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        #
        file = self.__save_fig(mode, path)
        return file

    def plot_roc(
            self,
            mode: str = None,
            path: str | Path = ".",
            figsize: tuple[int] = (4, 4),
    ) -> Path:
        """
        ROC curve for binary model

        Arguments
        ---------
        mode: str
            one of 'train' or 'test'.
        path: str | Path = "."
            directory where the file with plot will be written
            to not write the plot to file
            empty string "" may be passed (or anything which evaluates to `False` in `if` statement);
        figsize: tuple[int] = (4, 4)
            size of a figure in inches, (width, height).

        Returns
        -------
        file: str
            name of a file to which the plot is stored (in a current working directory);
            this name is set automatically based on data from self.storage.
            empty string (meaning the plot was not written to file) if `path=""`.
        """
        y, y_hat = self._yyhat(mode)
        plt.figure(figsize=figsize)
        #
        fpr, tpr, _ = roc_curve(y, y_hat)
        plt.plot(fpr, tpr)
        plt.axline((0, 0), (1, 1), color='gray', linewidth=.7)
        plt.title(f"ROC {mode}")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        #
        file = self.__save_fig(mode, path)
        return file

    def plot_calibration(
            self,
            mode: str = None,
            path: str | Path = ".",
            figsize: tuple[int] = (4, 4),
    ) -> Path:
        """
        Plots calibration curve for binary model.

        Arguments
        ---------
        mode: str
            one of 'train' or 'test'.
        path: str | Path = "."
            directory where the file with plot will be written
            to not write the plot to file
            empty string "" may be passed (or anything which evaluates to `False` in `if` statement);
        figsize: tuple[int] = (6, 6)
            size of a figure in inches, (width, height).

        Returns
        -------
        file: str
            name of a file to which the plot is stored (in a current working directory);
            this name is set automatically based on data from self.storage.
            empty string (meaning the plot was not written to file) if `path=""`.
        """
        y, y_hat = self._yyhat(mode)
        plt.figure(figsize=figsize)
        #
        prob_true, prob_pred = calibration_curve(y, y_hat, strategy="quantile", n_bins=10)

        plt.plot(prob_pred, prob_true, linewidth=2, marker='o')
        plt.xlabel('predicted prob', fontsize=14)
        plt.ylabel('empirical prob', fontsize=14)
        axis_lim = max(max(prob_pred), max(prob_true)) * 1.1
        plt.ylim((0, axis_lim))
        plt.xlim((0, axis_lim))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        names_legend = [f'model_{self.stage}', 'ideal']
        plt.legend(names_legend, fontsize=10)
        plt.title(f"calibration curve - {mode}")
        #
        file = self.__save_fig(f"calibration-{mode}", path)
        return file

    def plot_lift(
            self,
            mode: str,
            path: str | Path = ".",
            figsize: tuple[int] = (4, 4),
            nticks: int = 10,
            title: str = "lift",
            scikit: bool = True,
    ) -> str | Path:
        """"""
        fig = plt.figure(figsize=figsize)

        if not scikit:
            if mode in ["train", "test"]:
                ss = self.metrics.lift[mode].to_series(False)
                ss.index = ss.index.astype(str)
            else:
                raise ValueError(f'mode can be "train" or "test", given {mode}')

            q = len(ss)
            step = max(round(q / nticks), 1)
            xticks = list(range(0, q, step))
            if q - 1 not in xticks:
                xticks.append(q - 1)

            ax = ss.plot(xticks=xticks)
            ax.axhline(1, color='gray', linewidth=.7)
        else:
            y, y_hat = self._yyhat(mode, True)

            ax = fig.add_subplot(111)
            ax = plot_lift_curve(y, y_hat, ax=ax)

        ax.set_title(title + " " + mode)
        ax.set_ylabel(title)
        ax.set_xlabel("quantile")
        #
        file = self.__save_fig(f"lift-{mode}", path)
        return file

    def plot_gain(
            self,
            mode: str,
            path: str | Path = ".",
            figsize: tuple[int] = (4, 4),
            nticks: int = 10,
            title: str = "gain",
            scikit: bool = True,
    ) -> str | Path:
        """"""
        fig = plt.figure(figsize=figsize)

        if not scikit:
            if mode in ["train", "test"]:
                ss = self.metrics.gain[mode].to_series(False)
                ss.index = ss.index.astype(str)
            else:
                raise ValueError(f'mode can be "train" or "test", given {mode}')

            q = len(ss)
            step = max(round(q / nticks), 1)
            xticks = list(range(0, q, step))
            if q - 1 not in xticks:
                xticks.append(q - 1)

            ax = ss.plot(xticks=xticks)
            ax.axline((0, 0), (q - 1, 1), color='gray', linewidth=.7)

        else:
            y, y_hat = self._yyhat(mode, True)

            ax = fig.add_subplot(111)
            ax = plot_cumulative_gain(y, y_hat, ax=ax)     # figsize=figsize)

        ax.set_title(title + " " + mode)
        ax.set_ylabel(title)
        ax.set_xlabel("quantile")
        #
        file = self.__save_fig(f"gain-{mode}", path)
        return file

    def plot_precision_recall(
            self,
            mode: str,
            path: str | Path = ".",
            figsize: tuple[int] = (6, 6),
    ) -> str | Path:
        """"""
        y, y_hat = self._yyhat(mode)
        if mode == "train":
            p, r = self.metrics.precision_opt[mode], self.metrics.recall_opt[mode]
            label_pr = "optimal (precision, recall)"
        elif mode == "test":
            p, r = self.metrics.precision[mode], self.metrics.sensitivity[mode]
            label_pr = "obtained (precision, recall)"
        plt.figure(figsize=figsize)
        #
        print("Plot", p, r)
        precision, recall, _ = precision_recall_curve(y, y_hat)
        plt.plot(recall, precision, marker='.')
        plt.scatter([r], [p], marker='o', color='green')
        plt.title(f"precision-recall - {mode}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(["precision-recall curve", label_pr])
        #
        file = self.__save_fig(f"precrec-{mode}", path)
        return file

    def plot_score_distribution(
            self,
            mode: str,
            path: str | Path = ".",
            figsize: tuple[int] = (7, 7),
    ) -> str | Path:
        """"""
        y, y_hat = self._yyhat(mode)
        plot_covariates(
            y_hat, y,   # !
            varname="score", covarname="target",
            title_suffix=f" - {mode}",
            what=[["grouped_cloud", "distr"], ["boxplots", "densities"]],
            figsize=figsize,
        )
        #
        file = self.__save_fig(f"score-{mode}", path)
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
        plots: str,
            dictionary of file names (in a current working directory) for all generated plots;
       """

        plot_files = {}  # could be ordinary list but it's better managable (via names) for the future

        # If new threshold was set, then recalculate results
        if self.binary:
            self.performance(separately=self.separately)

        stats_df = self.metrics.to_frame().round(precision)

        if self.binary and isinstance(self.metrics, MetricsBin):

            if "confusion" in self.metrics._stats:
                confusion = self.metrics.confusion_df()
                confusion = confusion.reset_index().rename(columns={'level_0': ''})
                conf_train = confusion[confusion[''] == 'train']
                conf_test = confusion[confusion[''] == 'test']
                confusion_train = "__Confusion matrix__, train, threshold ="\
                                  + f"{round(self.metrics.threshold_opt['train'], 5)}"\
                                  + "\n\n" + conf_train.to_markdown(index=False)
                confusion_test = "__Confusion matrix__, test, threshold ="\
                                  + f"{round(self.metrics.threshold_opt['test'], 5)}"\
                                  + "\n\n" + conf_test.to_markdown(index=False)
                confusion = confusion_train + "\n\n" + confusion_test
            else:
                confusion = ''

            if "nobs" in self.metrics._stats:
                df_target = pd.DataFrame(
                    [[0, self.metrics.nobs0["train"], self.metrics.nobs0["test"]],
                     [1, self.metrics.nobs1["train"], self.metrics.nobs1["test"]]],
                    columns=["target", "train", "test"])
                stats_df.drop(index="nobs0", axis=1, inplace=True)
                stats_df.drop(index="nobs1", axis=1, inplace=True)
                nobs = "### Number of 0/1 in target for train and test sets" + "\n\n" + \
                    df_target.to_markdown(index=False)
            else:
                nobs = ''

            fig_train = self.plot_roc("train", path)
            fig_test = self.plot_roc("test", path)
            plot_files["roc_train"] = fig_train
            plot_files["roc_test"] = fig_test
            roc = "###" + "\n\n" + \
                f"![train]({fig_train})" + "\n\n" + \
                f"![test]({fig_test})"

            if "lift" in self.metrics._stats:
                # lift = self.metrics.lift_df()
                fig_lift_train = self.plot_lift("train", path)
                fig_lift_test = self.plot_lift("test", path)
                plot_files["lift_train"] = fig_lift_train
                plot_files["lift_test"] = fig_lift_test
                lift = "###" + "\n\n" + \
                    f"![train]({fig_lift_train})" + "\n\n" + \
                    f"![test]({fig_lift_test})"
            else:
                lift = ""

            if "gain" in self.metrics._stats:
                # gain = self.metrics.gain_df()
                fig_gain_train = self.plot_gain("train", path)
                fig_gain_test = self.plot_gain("test", path)
                plot_files["gain_train"] = fig_gain_train
                plot_files["gain_test"] = fig_gain_test
                gain = "###" + "\n\n" + \
                    f"![train]({fig_gain_train})" + "\n\n" + \
                    f"![test]({fig_gain_test})"
            else:
                gain = ""

            if 'precision' in self.metrics._stats or 'sensitivity' in self.metrics._stats:
                fig_precrec_train = self.plot_precision_recall("train", path)
                fig_precrec_test = self.plot_precision_recall("test", path)
                plot_files["precrec_train"] = fig_precrec_train
                plot_files["precrec_test"] = fig_precrec_test
                precrec = "###" + "\n\n" + \
                    f"![train]({fig_precrec_train})" + "\n\n" + \
                    f"![test]({fig_precrec_test})"
            else:
                precrec = ""

            fig_score_train = self.plot_score_distribution("train", path)
            fig_score_test = self.plot_score_distribution("test", path)
            plot_files["score_train"] = fig_score_train
            plot_files["score_test"] = fig_score_test
            score = "###" + "\n\n" + \
                f"![train]({fig_score_train})" + "\n\n" + \
                f"![test]({fig_score_test})"

            fig_calibration_train = self.plot_calibration("train", path)
            fig_calibration_test = self.plot_calibration("test", path)
            plot_files["calibration_train"] = fig_calibration_train
            plot_files["calibration_test"] = fig_calibration_test
            calibration = "###" + "\n\n" + \
                    f"![train]({fig_calibration_train})" + "\n\n" + \
                    f"![test]({fig_calibration_test})"

            stats = [confusion, nobs]
            plots = [roc, lift, gain, precrec, calibration, score]

        else:

            fig_train = self.plot_regression("train", path)
            fig_test = self.plot_regression("test", path)
            plot_files["reg_train"] = fig_train
            plot_files["reg_test"] = fig_test
            regression = "###" + "\n\n" + \
                f"![train]({fig_train})" + "\n\n" + \
                f"![test]({fig_test})"

            stats = []
            plots = [regression]

        if self.importances is not None:
            features_importances = "### Feature Importances" + "\n\n" + \
                self.importances.round(precision).to_markdown()
        else:
            features_importances = ""

        content = [
            f"# {self.storage.model_id} -- {self.storage.objective.upper()} -- {self.stage}",
            f"## Report {dt.datetime.now().isoformat()[:19]}",  # with/without Time Zone ?
            "### Statistics",
            stats_df.to_markdown(),
            *stats,
            features_importances,
            *plots,
            #
            "### Config",
            "```\n" + yaml.dump(self.config, sort_keys=False, indent=4) + "\n```"
        ]

        file = file if file else self.file_stem
        file = Path(path) / file
        file_out = file.with_suffix(".md")

        with open(file, "wt") as f:
            f.write("\n\n".join(content))

        return file_out, plot_files

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
            directory where the file with  feature importances will be written

        Returns
        -------
        file_out: Path,
            name of a file to which the feature importances is stored (in a current working directory).
        """
        if self.importances is not None:
            file = file if file else f"{self.file_stem}-feature_importances"
            file = Path(path) / file
            file_out = file.with_suffix(".json")
            importances_json = self.importances.to_json()
            with open(file_out, 'w') as f:
                json.dump(importances_json, f)
        else:
            file_out = ""
        return file_out

    def save_confusion_matrix(
            self,
            file: Union[str, PosixPath] = None,
            path: str | Path = ".",
    ) -> Path:
        """
        Arguments
        ---------
        file: str
            name of file to which confusion matrix shall be stored;
            if None then set automatically based on data from `self.storage`.
        path: str | Path = "."
            directory where the file with confusion matrix will be written

        Returns
        -------
        file_out: Path,
            name of a file to which the fconfusion matrix is stored (in a current working directory).
        """
        if self.binary:
            file = file if file else f"{self.file_stem}-confusion_matrix"
            file = Path(path) / file
            file_out = file.with_suffix(".csv")
            self.metrics.confusion_df().to_csv(file_out)
        else:
            file_out = ""
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

        file_md, plots = self.report_md(name, precision, path=path)
        file_pdf = self.report_pdf(path=path)

        file_metrics = self.save_metrics(path=path)
        file_importances = self.save_feature_importances(path=path)
        file_confusion_matrix = self.save_confusion_matrix(path=path)

        if storage:
            match storage:
                case True | 'copy':
                    fun = self.storage.copy
                case 'move':
                    fun = self.storage.move
                case _:
                    raise ValueError(f'Invalid storage file movement option; given `{storage}`.')

            fun(file_md, "report")

            for fig in plots:
                fun(plots[fig], "report")

            fun(file_pdf, "report")
            fun(file_metrics, "report")
            fun(file_importances, "report")
            fun(file_confusion_matrix, "report")
