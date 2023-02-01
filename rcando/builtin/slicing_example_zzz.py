#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 12:32:13 2023

@author: arek
"""

# %%
import yaml

import numpy as np
import pandas as pd

from .slicing import parse_slice, subseq, subseq_np, subseq_ss, subseq_pd, subseq_df

# %%
parse_slice(slice(1))
parse_slice(':1')
parse_slice(':-1')
parse_slice('-1')
parse_slice('~:-1')
parse_slice('3:')
parse_slice('~3:')

"""
%timeit parse_slice(slice(1))   # 145 ns ± 2.01 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
%timeit parse_slice('~(:-1)')   # 684 ns ± 10.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
%timeit parse_slice('~3:-2')    # 664 ns ± 6.36 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
"""

ll = list(range(10))
ll
subseq(ll, '~(:-1)')
subseq(ll, '~3:-2')
"""
%timeit subseq(ll, '~(:-1)')     # 1.65 µs ± 139 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
%timeit subseq(ll, '~3:-2')      # 1.5 µs ± 37.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
"""

npll = np.array(ll) + 10
subseq_np(npll, '~(:-1)')
subseq_np(npll, '~3:-2')
"""
%timeit subseq_np(npll, '~(:-1)')  # 17.8 µs ± 473 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
%timeit subseq_np(npll, '~3:-2')   # 13.9 µs ± 685 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
"""

pdll = pd.Series(ll) + 10
pdll.index = [2, 7, 1, 0, 9, 6, 3, 8, 5, 4]
pdll
subseq_ss(pdll, '~(:-1)')
subseq_pd(pdll, '~(:-1)')   # alias
subseq_ss(pdll, '~(:-1)', loc=True)  # KeyError: -1
subseq_ss(pdll, '~3:-2')
subseq_ss(pdll, '~3:-2', loc=True)   # KeyError: -2
subseq_ss(pdll, '~0:8')
subseq_ss(pdll, '~0:8', loc=True)
"""
%timeit subseq_ss(pdll, '~(:-1)')  # 176 µs ± 8.19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit subseq_ss(pdll, '~0:8')    # 169 µs ± 3.17 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit subseq_ss(pdll, '~0:8', loc=True)  # 187 µs ± 3.27 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
"""

dfll = pd.DataFrame(pdll)
subseq_df(dfll, '~(:-1)')
subseq_df(dfll, '~(:-1)', loc=True)  # KeyError: -1
subseq_df(dfll, '~3:-2')
subseq_df(dfll, '~0:8')
subseq_df(dfll, '~0:8', loc=True)
"""
%timeit subseq_df(dfll, '~(:-1)')  # 194 µs ± 3.29 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
%timeit subseq_df(dfll, '~0:8')    # 186 µs ± 2.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
%timeit subseq_df(dfll, '~0:8', loc=True)  # 198 µs ± 6.95 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
"""

# %%
# %%
N = 22
df = pd.DataFrame(
    {'a': np.random.randint(0, 2, size=N),
     'b': np.random.randint(0, 2, size=N),
     'c': np.random.randint(0, 9, size=N)}
    )
df = df.sort_values(['a', 'b', 'c'], ascending=True)
df

# %%
dfg = df.groupby(['a', 'b'])
df['cc'] = dfg.cumcount()
dfg.apply(lambda df: df.iloc[slice(1,-1)])
dfg.apply(lambda df: df[slice(1,-1)]).index.get_level_values(2)

# %%
# %%
config = yaml.load(open("common/builtin/slicing_example.yaml", "r"), Loader=yaml.FullLoader)
etm = config["ecom_target_mode"]
etm

dfg.apply(lambda df: subseq_df(df, '~1:-1'))
dfg.apply(lambda df: df.loc[subseq(df.index.tolist(), '~1:-1'), :])

"""
%timeit dfg.apply(lambda df: subseq_df(df, '~1:-1'))
    # 2.4 ms ± 59.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit dfg.apply(lambda df: df.loc[subseq(df.index.tolist(), '~1:-1'), :])
    # 2.6 ms ± 63.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
"""

# %%
