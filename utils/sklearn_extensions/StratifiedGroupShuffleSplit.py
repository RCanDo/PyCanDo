#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- link: https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit
- link: https://github.com/scikit-learn/scikit-learn/issues/12076
"""
import numpy as np
import pandas as pd
import random


class StratifiedGroupShuffleSplit(object):
    """
    This is primitive version of stratified grouped shuffle split:

    1. Only binary, 0-1 valued target `y` is considered;
    more precisely, all values different from 1 are considered as one class, say non-1,
    hence one may pass whatever variable it likes but the algorithm only distinguishes values 1 and non-1.

    2. Splits are made wrt. unique group values, i.e.
    `test_size` proportion is applied to number of unique values of a group variable.
    Doing it wrt. the exact number of observations (records) is much more problematic and rather less useful.

    3. Stratification is also done wrt unique group values, i.e.
    for a given unique group value if there is at least one observation within this group where `y = 1`
    then it is considered that `y = 1` for the whole group.
    Thus each unique group value is unambiguously ascribed to one of `y = 1` class or "y != 1" class.
    Then from each of these classes the `test_size` portion of unique group values is randomly chosen
    and put into "test" portion of the data.
    The rest is "train" portion of unique group values and consists of `1 - test_size` unique group values
    from each of the `y = 1` class or "y != 1" class.
    """

    def __init__(
            self,
            n_splits: int,
            test_size: float | int,
            train_size: float | int = None,
            random_state: int = None,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tran_size = train_size     # not used
        self.random_state = random_state
        # np.random.seed(random_state)

    def _split_portion(
            self,
            items: list | np.ndarray,
            portion: float,
    ) -> tuple[np.array, np.array]:
        """
        Splits set of `items` into two subsets, say `items_train` and `items_test`,
        with proportions `1 - portion`, `portion` respectively.
        ! It is assumed that elements of `items` are unique.
        If not, the algorithm will return only unique elements of `items`
        and each will belong to only one of the "train" or "test" subsets.
        """
        n = len(items)
        n_test = int(np.ceil(n * portion))   # at least 1
        random.seed(self.random_state)
        items_test = random.sample(sorted(items), n_test)
        items_train = set(items) - set(items_test)
        return np.array(list(items_train)), items_test

    def split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[np.array, np.array]:
        """
        `X`, `y`, `groups` must all have the same length.
        Returns pair of np.arrays of  numeric indices  (!)
        (those refered to via .iloc method of pd.DataFrame or pd.Series)
        of `X`, `y` and `groups` giving split of theses data with proportions `1 - test_size`, `test_size`,
        however wrt unique values of `groups` variable.
        """
        idx = np.arange(len(y))

        if isinstance(self.test_size, int):
            test_size = self.test_size / len(y)
        else:
            test_size = self.test_size

        gu = groups.unique()
        gu_1 = groups[y == 1].unique()
        gu_0 = list(set(gu) - set(gu_1))

        i = 1
        while i <= self.n_splits:

            gu_1_train, gu_1_test = self._split_portion(gu_1, test_size)
            gu_0_train, gu_0_test = self._split_portion(gu_0, test_size)

            gu_train = np.r_[gu_1_train, gu_0_train]
            gu_test = np.r_[gu_1_test, gu_0_test]

            idx_train = idx[groups.isin(gu_train)]
            idx_test = idx[groups.isin(gu_test)]

            i += 1

            yield idx_train, idx_test
