import numpy as np
# import pandas as pd
import datetime as dt
from numpy.typing import NDArray
from typing import Union, Literal
from collections.abc import Any, Sequence, Iterable  # List, Tuple, Callable, Optional, MutableMapping

Int = Union[int, np.integer]
Float = Union[float, np.floating]
Number = Union[int, float, np.number]  # ?== Union[Int, Float]
Numeric = Union[Sequence[Number], NDArray[np.number]]  # , 'pd.Series[float]']
Factor = Union[Sequence[str], NDArray[np.str_]]  # , 'pd.Series[str]']
Variable = Union[Numeric, Factor]

# Notes
# =====
#
#  np.number, np.floating, np.integer
#  ----------------------------------
# isinstance(np.float64(3), np.number)    # True
# isinstance(np.float64(3), np.floating)  # True
# isinstance(np.float64(3), np.integer)   # False
# isinstance(np.int64(3), np.number)      # True
# isinstance(np.int64(3), np.floating)    # False
# isinstance(np.int64(3), np.integer)     # True
# np.floating(3)  # TypeError: cannot create 'numpy.floating' instances
# np.integer(3)   # TypeError: cannot create 'numpy.integer' instances
#
#  Sequence / Iterable / ...
#  -------------------------
# https://stackoverflow.com/questions/72157296/what-is-the-difference-between-type-hinting-a-variable-as-an-iterable-versus-a-s
# The Sequence and Iterable abstract base classes (can also be used as type annotations)
# mostly* follow Python's definition of sequence and iterable. To be specific:
# * Iterable is any object that defines __iter__ or __getitem__.
# * Sequence is any object that defines __getitem__ and __len__.
#       By definition, any sequence is an iterable.
#       The Sequence class also defines other methods such as __contains__, __reversed__
#       that calls the two required methods.
# Some examples:
# – list, tuple, str are the most common sequences.
# – Some built-in iterables are not sequences.
#   For example, `reversed` returns a reversed object (or `list_reverseiterator` for lists)
#   that cannot be subscripted.
#
# Iterable does not exactly conform to Python's definition of iterables
# — it only checks if the object defines __iter__,
# and does not work for objects that's only iterable via __getitem__
# (see this table for details: https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes).
# The gold standard of checking if an object is iterable is using the `iter()` builtin.
#
#   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
#               | Sequence  | Iterable  | Sized ^1  | SupportsIndex ^2
# list              +           +           +         -!
# tuple             +           +           +         -!
# set               -           +           +         -!
# dict              -           +           +         -!
# range             +           +           +         -
# generator         -           +           -         -
# pd.Series         -           +           +         +
# np.array          -           +           +         +
# str               +           +           +         -
#
# len(Iterable)     # no!  :/  by mypy
# len(Sequence)     # ok
#
# ^1 collections.abc.Sized  checks for .__len__()
# ^2 typing.SupprtsIndex  checks for .__index__()
#    – not what you think, see: https://stackoverflow.com/questions/27449310/python-index-special-method
# -----------------------------------------------------------------------------
# df.index(pd.Index)                # ! i.e. only pd.Index "family"
# df.index.intersection(pd.Index)   #   "
#
#  numpy typing shape
#  ------------------
# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype

ValuesStrict = NDArray[np.number]
Values = Union[ValuesStrict, Sequence[Number], Iterable[Number]]

PositionalIndexPoint = np.integer
PositionalIndexStrict = NDArray[PositionalIndexPoint]
PositionalIndex = Union[PositionalIndexStrict, Sequence[PositionalIndexPoint]]
IndexPoint = Union[np.integer, np.datetime64, np.str_]  # dt.time, dt.datetime,
IndexStrict = NDArray[IndexPoint]
Index = Union[IndexStrict, Sequence[IndexPoint]]
# should be rather  Index  and  Indices  but we adhere to pandas convention of using "index" for "indices";

Length = Int
Width = Union[Number, dt.timedelta]
IntegralType = Literal["trapezoid", "simpson"]
AreaUnderPeak = Number
Areas = dict[IntegralType, AreaUnderPeak]


def iterable_not_str(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, bool)
