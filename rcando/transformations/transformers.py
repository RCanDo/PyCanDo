#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Transformators according to instructions
version: 1.0
type: module
keywords: [transformer, variables, instructions]
description: |
    Variables transformers according to instructions from some config lists/dicts/etc.
remarks:
    - etc.
todo:
    - problem 1
sources:
file:
    usage:
        interactive: False   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False      # if the file is intended to be run in a terminal
    date: 2022-02-10
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import sys
sys.path.insert(1, "../")

# import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.exceptions import NotFittedError


# %%
def process_ss(ss, np_name="from_numpy_array"):
    try:
        shape = ss.shape
    except AttributeError:
        shape = (1,)

    if len(shape) > 1:
        sname = np_name
        idx = range(shape[0])
    else:
        ss = pd.Series(ss)
        idx = ss.index
        sname = ss.name
        ss = ss.to_numpy().reshape(-1, 1)

    return ss, sname, idx


def from_sklearn_inverse(transformer, name="T_inverse"):
    """
    transformer: sklearn transformer with .inverse_transform(), .transform() and .fit_transform() methods;
    name: name to be given to standardised version of a transformer.inverse_transform();

    Returns `standardised` version of a sklearn `transformer.inverse_transform()` i.e.
    which works as ordinary function -- not as class with methods.
    Instead of running
    `transformer.inverse_transform(y)` where `y` is necessarily np.array
    one may now run
    `standardised(x)` where `y` may be also a pd.Series for which the result will retain its index.

    `standardised` return pd.Series
    """
    t_inverse = transformer.inverse_transform

    # make it idempotent
    if t_inverse.__qualname__ == 'from_sklearn_inverse.<locals>.standardised':
        standardised = t_inverse

    else:
        if not name:
            name = transformer.__name__ + "_inverse"

        def standardised(ss):
            ss, sname, idx = process_ss(ss)

            ss = t_inverse(ss)[:, 0]

            ss = pd.Series(ss, index=idx)
            ss.name = sname
            # t_inverse.__func__.__name__ = name
            # ?  is this possible to make it more informative e.g. via .get_params()  ?

            return ss

        standardised.__name__ = name

    return standardised


# more general
def from_sklearn(transformer, name="T"):
    """
    transformer: sklearn transformer with .transform() and .fit_transform() methods;
    name: name to be given to standardised version of a transformer;

    Returns `standardised` version of a sklearn `transformer` i.e.
    which works as ordinary function - not as class with methods.
    Instead of running
    `transformer.transform(x)` or `transformer.fit_transform(x)`
    where `x` is necessarily np.array
    one may now run
    `standardised(x)` where `x` may be also a pd.Series for which the result will retain its index.

    Moreover `standardised` inherits `.inverse_transform()` from fitted `transformer`
    as its method, hence one may run `standardised.inverse_transform(y)` to get `x` back,
    where `y = standasrised(x)`.

    ! However, `standardised(x)` returns tuple:
    (y, standardised_fitted)
    where `y` is pd.Series with a value returned from `transformer.fit_transform(x)`
    and `standardised_fitted` is `standardised` with all params fitted.

    """

    # make it idempotent
    if hasattr(transformer, "__qualname__"):
        # it's a trick: we should rather check
        #  transformer.__qualname__ == 'from_sklearn.<locals>.standardised'
        # but sklearn transformers just don't have `__qualname__` attribute
        # so it's enough to check it's existence
        standardised = transformer

    else:
        def standardised(ss):
            ss, sname, idx = process_ss(ss)

            try:
                ss = transformer.transform(ss)[:, 0]
            except NotFittedError:
                print(f"! fitting parameters for {name} on variable {sname}")
                ss = transformer.fit_transform(ss)[:, 0]

            ss = pd.Series(ss, index=idx)
            ss.name = sname
            # transformer.__name__ = name
            # ?  is this possible to make it more informative e.g. via .get_params()  ?

            return ss, from_sklearn(transformer, name)

        standardised.__name__ = name
        standardised.inverse_transform = from_sklearn_inverse(transformer, name + "_inverse")

    return standardised


def power_transformer(ss: pd.Series):
    """
    Version of from_sklearn() dedicated to sklearn.PowerTransformer solely,
    only to have a good name of the function (used later in some important plots).
    """
    transformer = PowerTransformer()

    ss, sname, idx = process_ss(ss)

    try:
        ss = transformer.transform(ss)[:, 0]
    except NotFittedError:
        print(f"! fitting parameters for PowerTransformer on variable {sname}")
        ss = transformer.fit_transform(ss)[:, 0]

    ss = pd.Series(ss, index=idx)
    ss.name = sname

    t_name = transformer.get_params()['method'].title().replace("-", "") + \
        "_{{\\lambda = {}}}".format(round(transformer.lambdas_[0], 2))

    return ss, from_sklearn(transformer, t_name)


# %%
"""
ss = data_raw['minMidpointDistanceStd']
ss = data_raw.loc[:, ['minMidpointDistanceStd']]
ss.shape
ss.head()

powtrans = from_sklearn(PowerTransformer)
zz, transformer = powtrans(ss)
transformer.__name__

zz, transformer = power_transformer(ss)
transformer.__name__
"""


# %%
def transform(
        data: pd.DataFrame,
        transformations: dict(),
        inverse: bool = False, ):
    """"""
    columns = set(data.columns).intersection(transformations.keys())
    data_0 = data.copy()

    for c in columns:

        item = transformations[c]
        if any(item.values()):
            ss = data_0.pop(c)
            ss = ss.dropna()
            if inverse:
                if item["lower_t"]:
                    ss = ss[ss >= item["lower_t"]]
                if item["upper_t"]:
                    ss = ss[ss <= item["upper_t"]]

                # transformation inverse
                if item["inverse"]:
                    ss = item["inverse"](ss)

                if item["lower"]:
                    ss = ss[ss >= item["lower"]]
                if item["upper"]:
                    ss = ss[ss <= item["upper"]]

            else:
                if item["lower"]:
                    ss = ss[ss >= item["lower"]]
                if item["upper"]:
                    ss = ss[ss <= item["upper"]]

                # transformation
                if item["forward"]:
                    try:
                        # for transformers crated with  from_sklearn()  (see above)
                        ss, transformer = item["forward"](ss)
                        item['forward'] = transformer
                        if not item["inverse"]:
                            item["inverse"] = from_sklearn_inverse(transformer)
                            # from_sklearn_inverse()  is idempotent so it's safe
                            # although  `... = transformer.inverse_transform`  would more straightforward;
                            # result is exactly the same!
                    except Exception:
                        ss = item["forward"](ss)

                if item["lower_t"]:
                    ss = ss[ss >= item["lower_t"]]
                if item["upper_t"]:
                    ss = ss[ss <= item["upper_t"]]

            data_0 = pd.concat([data_0, ss], axis=1)

    return data_0, transformations


# %%
