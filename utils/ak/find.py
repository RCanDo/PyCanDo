#%%
#
def find_0(lst, value):
    res = [i for i, v in enumerate(lst) if v==value]
    return res

#%%
#
def find_all(lst: list, value) -> list[int]:
    """
    Find all occurances of a `value` in a list `lst`;
    Returns list of all __indices__ of a `value` in a `lst`;
    If `value` is not present in `lst` then returns empty list.
    E.g.
     find_all([1, 3, 2, 4, 2, 5, 3, 4, 2], 2) -> [2, 4, 8]
     find_all([1, 3, 2, 4, 2, 5, 3, 4, 2], 33) -> []
    This method is faster then list comprehension:
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

#%%
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

#%%
#%%
import numpy as np
data = np.random.choice([0, 1], 100, replace=True).tolist()
data

%timeit find_0(data, 1)
# 7.05 µs ± 178 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
%timeit find_all(data, 1)
# 11.7 µs ± 73.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

#%%
data = np.random.choice([0, 1], 100, replace=True, p=[.9, .1]).tolist()
data

%timeit find_0(data, 1)
# 5.89 µs ± 295 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
%timeit find_all(data, 1)
# 3.48 µs ± 85 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%timeit find_0(data, 0)
# 7.59 µs ± 390 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
%timeit find_all(data, 0)
# 17.8 µs ± 1.62 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

#%%
data = np.random.choice([0, 1], 10000, replace=True).tolist()

%timeit find_0(data, 1)
# 723 µs ± 14.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit find_all(data, 1)
# 1.11 ms ± 52.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

#%%
data = np.random.choice([0, 1], 10000, replace=True, p=[.9, .1]).tolist()

%timeit find_0(data, 1)
# 678 µs ± 34 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit find_all(data, 1)
# 294 µs ± 4.33 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

%timeit find_0(data, 0)
# 796 µs ± 16 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit find_all(data, 0)
# 1.79 ms ± 46.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

#%% WHAT ABOUT filter ???

#%%
