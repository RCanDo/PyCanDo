#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import (ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit,  # type: ignore # [import]
                                     StratifiedGroupKFold, StratifiedKFold, GroupKFold,
                                     KFold)  # type: ignore # [import]

from .sklearn_extensions import StratifiedGroupShuffleSplit

from typing import Union, Optional, Callable, Iterator, Literal, cast, get_args
from numpy.typing import NDArray
from numpy import int_

ModelReg = Literal["continuous", "regression", "reg", "r", "val"]
ModelClass = Literal["classifier", "class", "categorical", "cat", "binary", "bin", "b"]
ModelType = Union[ModelReg, ModelClass]
TrainTestSplitIndices = tuple[NDArray[int_], NDArray[int_]]
TrainTestSplitData = list[Union[pd.DataFrame, "pd.Series[float]"]] | dict[str, Union[pd.DataFrame, "pd.Series[float]"]]

SPLIT_KEYS = {"X_train", "X_test", "X_valid", "y_train", "y_test", "y_valid"}


def get_cv_split_generator(
        X: pd.DataFrame, y: pd.Series[float],
        model_type: ModelType,
        cv_folds: int,
        groups: Optional[pd.Series[float]] = None
) -> Iterator[TrainTestSplitIndices]:
    """
    Returns cross-validation split generator, i.e. from _KFold group
    """
    if model_type in get_args(ModelReg):
        if groups is not None:
            splitter = GroupKFold(n_splits=cv_folds).split(X, y, groups)
        else:
            splitter = KFold(n_splits=cv_folds).split(X, y)
    elif model_type in get_args(ModelClass):
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=cv_folds).split(X, y, groups)
        else:
            splitter = StratifiedKFold(n_splits=cv_folds).split(X, y)
    else:
        model_reg = ", ".join(f"'{r}'" for r in get_args(ModelReg))
        model_class = ", ".join(f"'{c}'" for c in get_args(ModelClass))
        raise ValueError(f"`objective` must be one of {model_reg} \nor one of {model_class}.")
    return cast(Iterator[TrainTestSplitIndices], splitter)


def get_shuffle_split_generator(
        X: pd.DataFrame, y: pd.Series | pd.DataFrame,
        model_type: ModelType,
        test_size: float | int = .3,
        groups: Optional[pd.Series] = None,
        seed: int = 1234
) -> Iterator[TrainTestSplitIndices]:
    """
    Returns shuffle split generator, i.e. from _ShuffleSplit group
    """
    if model_type in get_args(ModelReg):
        if groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups)
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y)
    elif model_type in get_args(ModelClass):
        if groups is not None:
            if isinstance(y, pd.DataFrame):
                raise Exception(
                    "If `model_type` is one of 'binary', 'categorical', etc.\n"
                    "and `groups` are not empty then `y` cannot be pd.DataFrame – must be pd.Series."
                )
            splitter = StratifiedGroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups)
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y)
    else:
        model_reg = ", ".join(f"'{r}'" for r in get_args(ModelReg))
        model_class = ", ".join(f"'{c}'" for c in get_args(ModelClass))
        raise ValueError(f"`objective` must be one of {model_reg} \nor one of {model_class}.")
    return cast(Iterator[TrainTestSplitIndices], splitter)


def shuffle_split(
        X: pd.DataFrame, y: pd.Series | pd.DataFrame,
        model_type: ModelType,
        test_size: float | int = .3,
        groups: Optional[pd.Series] = None,
        seed: int = 1234
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Returns data split into "train" and "test" parts in proportions `1 - test_size` and `test_size`.
    """
    splitter = get_shuffle_split_generator(X, y, model_type, test_size, groups, seed)
    idx_train, idx_test = next(splitter)

    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test]

    return X_train, X_test, y_train, y_test


def train_test_valid_split(
        df: pd.DataFrame,
        model_type: ModelType,
        valid_size: float = 0.3,
        test_size: float = 0.3,
        features: Optional[list | None] = None,
        target: str | list[str] = "target",
        seed: int = 1234,
        as_dict: bool = True,
        groups: Optional[str] = None,
        transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> TrainTestSplitData:
    """
    First splits `df` into _train0 and _valid in proportions (1 - valid_size, valid_size).
    Then splits _train0 again into _train and _test in proportions (1 - test_size, test_size).
    I.e. `test_size` refers to (1 - valid_size) portion of the whole `df`.

    To keep records with the same value of some variable in the same dataset part
    pass the name of this variable to `groups`;
    then `test_size` and `valid_size` are applied to set of unique values of `groups`.

    If `model_type` is one of 'binary', 'categorical', etc.
    and `groups` are not empty then `target` cannot be list of column names – must be only one name (str).
    """
    if model_type in get_args(ModelClass) and groups is not None and isinstance(target, list):
        raise Exception(
            "If `model_type` is one of 'binary', 'categorical', etc.\n"
            "and `groups` are not empty then `target` cannot be list of column names – must be only one name (str)."
        )

    if transform is not None:
        df = transform(df)

    y = df[target]
    X = df[features] if features else df.drop(target, axis=1)
    group_var = X[groups] if groups else None

    if valid_size > 0:
        X_train0, X_valid, y_train0, y_valid = \
            shuffle_split(X, y, model_type, valid_size, group_var, seed)
    else:
        X_train0, X_valid, y_train0, y_valid = \
            X, X.iloc[:0], y, y.iloc[:0]    # type: ignore  [assignment]

    X_train, X_test, y_train, y_test = \
        shuffle_split(X_train0, y_train0, model_type, test_size, group_var, seed)

    print("train shape: ", X_train.shape)
    print("test  shape: ", X_test.shape)
    print("valid shape: ", X_valid.shape)

    result: TrainTestSplitData = [X_train, X_test, X_valid, y_train, y_test, y_valid]
    if as_dict:
        result = dict(
            X_train=X_train, X_test=X_test, X_valid=X_valid,
            y_train=y_train, y_test=y_test, y_valid=y_valid,
        )
        assert set(result.keys()) == SPLIT_KEYS
    return result
