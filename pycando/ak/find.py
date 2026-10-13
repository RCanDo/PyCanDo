from typing import Any


# %%
def find_all_0(lst: list[Any], value: Any) -> list[int]:
    """
    Find all occurances of a `value` in a list `lst`;
    Returns list of all __indices__ of a `value` in a `lst`;
    If `value` is not present in `lst` then returns empty list.
    E.g.
     find_all_0([1, 3, 2, 4, 2, 5, 3, 4, 2], 2)       # [2, 4, 8]
     find_all_0([1, 3, 2, 4, 2, 5, 3, 4, 2], 33)      # []
    This method is slower then `find_all()`
    for cases when `value` is rare (sparse) in a list.
    Otherwise it is faster.
    """
    res = [i for i, v in enumerate(lst) if v==value]
    return res


find_all_common = find_all_0  # alias

# %%
def find_all(lst: list[Any], value: Any) -> list[int]:
    """
    Find all occurances of a `value` in a list `lst`;
    Returns list of all __indices__ of a `value` in a `lst`;
    If `value` is not present in `lst` then returns empty list.
    E.g.
     find_all([1, 3, 2, 4, 2, 5, 3, 4, 2], 2)       # [2, 4, 8]
     find_all([1, 3, 2, 4, 2, 5, 3, 4, 2], 33)      # []
    This method is faster then `find_all_0()`, i.e. list comprehension:
     [i for i, v in enumerate(lst) if v == value]        # (1)
    for cases when `value` is rare (sparse) in a list.
    Otherwise list comprehension is faster.
    """
    i = 0
    idx = []
    while i >= 0:
        try:
            i = lst.index(value, i)
            idx.append(i)
            i += 1
        except ValueError:
            i = -1
    return idx


find_all_rare = find_all  # alias


# %%
"""
Examples
--------

ll = [1, 3, 2, 4, 2, 5, 3, 4, 2]
help(ll.index)
ll.index(2)         # 2
# find index of first 2 between 2nd position (inclusive, counting from 0) and 4th pos (exclusive)
ll.index(2, 2, 4)   # 2
ll.index(2, 3, 4)   # ValueError: 2 is not in list
ll.index(2, 3, 5)   # 4

print(find_all(ll, 2))
print(find_all(ll, 3))
print(find_all(ll, 33))

# %%
import numpy as np

# 1. small data

# 1.1 common occurances
data = np.random.choice([0, 1], 100, replace=True).tolist()
data

%timeit find_all_0(data, 1)     # 3.23 μs ± 123 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit find_all(data, 1)       # 4.59 μs ± 22.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# 1.2 rare occurances
data = np.random.choice([0, 1], 100, replace=True, p=[.9, .1]).tolist()
data

%timeit find_all_0(data, 1)     # 2.54 μs ± 60 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit find_all(data, 1)       # 1.99 μs ± 31.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# 1.3 very common (majority) occurances
%timeit find_all_0(data, 0)     # 3.25 μs ± 24.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
%timeit find_all(data, 0)       # 5.78 μs ± 31.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# 2. large data

# 2.1 common occurances
data = np.random.choice([0, 1], 10000, replace=True).tolist()

%timeit find_all_0(data, 1)     # 424 μs ± 5.97 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
%timeit find_all(data, 1)       # 478 μs ± 2.37 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# 2.2 rare occurances
data = np.random.choice([0, 1], 10000, replace=True, p=[.9, .1]).tolist()

%timeit find_all_0(data, 1)     # 370 μs ± 4.42 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
%timeit find_all(data, 1)       # 179 μs ± 2.01 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

# 2.3 very common (majority) occurances
%timeit find_all_0(data, 0)     # 403 μs ± 2.33 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
%timeit find_all(data, 0)       # 733 μs ± 18 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# 2.4 very rare occurances
data = np.random.choice([0, 1], 10000, replace=True, p=[.99, .01]).tolist()

%timeit find_all_0(data, 1)     # 354 μs ± 2.05 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
%timeit find_all(data, 1)       # 110 μs ± 1.3 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

# %% WHAT ABOUT filter ???
"""
