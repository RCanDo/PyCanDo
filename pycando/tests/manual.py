#! python3
# -*- coding: utf-8 -*-

# %%
from pycando.config import pandas_options
pandas_options()

# %%
# from .find
from pycando.ak import find_all_0, find_all  # , find_all_common, find_all_rare (aliases)
# from .math
from pycando.ak import power_series, power_weights  # , powser (alias)
# from .nppd
from pycando.ak import which, probability, center, data_frame, print_df  # , p, prob (aliases)
# from .random
from pycando.ak import htr, randkasp1  # , htrandint, hashtimerandom, hashtimerandint (aliases)

# %% ak
help(find_all_0)
help(find_all)
find_all([1, 3, 2, 4, 2, 5, 3, 4, 2], 2)       # [2, 4, 8]
find_all([1, 3, 2, 4, 2, 5, 3, 4, 2], 33)      # []

# %%
help(power_series)
help(power_weights)

# %%
help(which)
help(probability)
help(center)
help(data_frame)
help(print_df)

data_frame(range(66), range(22))
# %%
import numpy as np
import pandas as pd

df = pd.DataFrame(
    np.random.randint(3, size=12).reshape(3,4),
    columns=list("abcd")
    )
print_df(df)

# %%
help(htr)
help(randkasp1)

# %%
# from .builtin
from pycando.builtin import all_not_none, coalesce0, coalesce, \
	filter_first, dict_default, \
    partial_match, first_match, all_matches, \
	timeit, where_up, replace_deep, \
    dict_depth, adaptive_round, \
    fill_from_str, forward_fill, ffill, backward_fill, bfill, \
    lengthen, \
    align, align_lengthen, align_prod, align_cycle, \
    flatten, paste, union, \
    dict_set_union_0, dict_set_union, \
    dict_list_sum_0, dict_list_sum

# %%
help(all_not_none)
help(coalesce0)
help(coalesce)
help(coalesce)
help(filter_first)

# %%
# from .files
from pycando.builtin import file_nrows
# from .printing
from pycando.builtin import section, not_too_long, indent, show, iprint, Repr, repr
# from .slicing
from pycando.builtin import boolean_slice, parse_slice, subseq, subseq_np, subseq_ss, subseq_pd, subseq_df
# from .str
from pycando.builtin import filter_str, filter_re, iterprint, iter_print
# from .timer
from pycando.builtin import Times, DTimes, Timer
#from .process_text
from pycando.builtin import process_text
