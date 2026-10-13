import numpy as np
import pandas as pd
from typing import Iterable


def sample_dates(
        start: pd.Series | pd.Timestamp | str,
        end: pd.Series | pd.Timestamp | str,
        n_samples: int = 1,
        add_start: bool | str = False,
        add_end: bool | str = False,
        as_lists: bool = False,
        seed: int = 1234,
) -> 'pd.DataFrame[pd.Timestamp] | pd.Series[list[pd.Timestamp]]':   # type: ignore # [type-arg]
    """
    Randomly sample `n_samples` dates from the interval between `start` and `end`.
    If `start` and `end` are both Iterables (except strings) then they must be of equal length.
    Moreover indices of `start` and `end` are assumed to be aligned if present
    (if they are not aligned, i.e. sets of their values differ, then the result is wrong!).
    Also, both `start` and `end` may be a single value of datetime in the format understandable by pandas
    – then it is replicated to the length of the other (if the other is Iterable except string).
    Finally, it is strongly assumed that `start < end`
    – `NaT`s are returned for each `start >= end` pair.

    Arguments
    ---------
    start: pd.Series | pd.Timestamp | str,
        date or series of dates in date format understandable by pandas.
    end: pd.Series | pd.Timestamp | str
        date or series of dates in date format understandable by pandas.
    n_samples: int = 1,
        number of dates to sample for each `start`–`end` pair;
    add_start: bool | str = False,
        if True then `date_start` will be added as first column;
        if non-empty string then this column will get this name;
        empty string works as False.
    add_end: bool | str = False,
        if True then `date_end` will be added as last column;
        if non-empty string then this column will get this name;
        empty string works as False.
    as_lists: bool = False,
        if False (default) then DataFrame is returned of the size `n x n_samples`
        (`n` is the length of `start` and/or `end`)
        where each column is for one date sample;
        if True then each record of this dataframe is turned into list and then
        pd.Series of such lists is returned
        (all of this even for single valued `start` and `end` and `n_samples = 1`).
    seed: int = 1234,
        seed of random number generator passed to numpy.random module.

    Returns
    -------
    pd.DataFrame[pd.Timestamp] | pd.Series[list[pd.Timestamp]]:
        pd.Series of lists of dates or pd.DataFrame of dates.

    Examples
    --------
    res = sample_dates(['2022-01-01', '2022-01-10'], '2022-02-22', 3)
    type(res)   # pandas.core.frame.DataFrame
    res
    #            0          1          2
    # 0 2022-02-17 2022-01-20 2022-02-08
    # 1 2022-01-22 2022-02-03 2022-01-25
    res.shape   # (2, 3)

    res = sample_dates(['2022-01-01', '2022-01-10'], '2022-02-22', 3, as_lists=True)
    type(res)   # pandas.core.series.Series
    res
    # 0    [2022-02-17 00:00:00, 2022-01-20 00:00:00, 2022-02-08 00:00:00]
    # 1    [2022-01-22 00:00:00, 2022-02-03 00:00:00, 2022-01-25 00:00:00]
    # dtype: object
    res.to_list()
    # [[Timestamp('2022-02-17 00:00:00'), Timestamp('2022-01-20 00:00:00'), Timestamp('2022-02-08 00:00:00')],
    #  [Timestamp('2022-01-22 00:00:00'), Timestamp('2022-02-03 00:00:00'), Timestamp('2022-01-25 00:00:00')]]

    sample_dates('2022-01-01', '2022-02-22')
    #            0
    # 0 2022-02-17          # DataFrame

    sample_dates('2022-01-01', '2022-02-22', as_lists=True)
    # 0    [2022-02-17 00:00:00]
    # dtype: object         # Series

    sample_dates(['2022-01-01', '2022-01-10'], ['2022-02-22', '2022-01-10'], 3)
    #            0          1          2
    # 0 2022-02-17 2022-01-20 2022-02-08
    # 1        NaT        NaT        NaT

    sample_dates(pd.Series(['2022-01-01', '2022-01-10']), pd.Series(['2022-02-22', '2022-01-11']), 3)
    #            0          1          2
    # 0 2022-02-17 2022-01-20 2022-02-08
    # 1 2022-01-10 2022-01-10 2022-01-10

    sample_dates(pd.Series(['2022-01-01', '2022-01-10']),
                 pd.Series(['2022-02-22', '2022-01-11'], index=[1,2]), 3)
    # ! IndexError: Indices of `start` and `end` are assumed to be aligned if present.
    #   If they are not aligned (i.e. sets of their values differ) then the result is wrong!

    sample_dates(pd.Series(['2022-01-01', '2022-01-10'], index=[2, 1]),
                 pd.Series(['2022-02-22', '2022-01-11'], index=[1, 2]), 3)
    # ok

    sample_dates(['2022-01-01', '2022-01-10'], ['2022-02-22'], 3)
    # ! ValueError: If `start` and `end` are both Iterables they must be of equal length.

    sample_dates(['2022-01-01', '2022-01-10'], ['2022-02-22', '2022-01-10'], 3, add_start=True)
    #   date_start          0          1          2
    # 0 2022-01-01 2022-02-17 2022-01-20 2022-02-08
    # 1 2022-01-10        NaT        NaT        NaT

    sample_dates(['2022-01-01', '2022-01-10'], ['2022-02-22', '2022-01-10'], 3, add_start='start')
    #        start          0          1          2
    # 0 2022-01-01 2022-02-17 2022-01-20 2022-02-08
    # 1 2022-01-10        NaT        NaT        NaT

    sample_dates(['2022-01-01', '2022-01-10'], ['2022-02-22', '2022-01-10'], 3, add_end=True)
    #            0          1          2   date_end
    # 0 2022-02-17 2022-01-20 2022-02-08 2022-02-22
    # 1        NaT        NaT        NaT 2022-01-10

    sample_dates(['2022-01-01', '2022-01-10'], ['2022-02-22', '2022-01-10'], 3, add_end='end')
    #            0          1          2        end
    # 0 2022-02-17 2022-01-20 2022-02-08 2022-02-22
    # 1        NaT        NaT        NaT 2022-01-10
    """

    if isinstance(start, Iterable) and not isinstance(start, str) \
      and isinstance(end, Iterable) and not isinstance(end, str) \
      and len(start) != len(end):
        raise ValueError(
            "If `start` and `end` are both Iterables they must be of equal length \n"
            "and must have the same set of indices i.e. they must be aligned."
        )

    start: pd.Series | pd.DatetimeIndex | pd.Timestamp = pd.to_datetime(start)
    end: pd.Series | pd.DatetimeIndex | pd.Timestamp = pd.to_datetime(end)
    intervals = pd.Series((end - start))  # type: ignore # [operator]
    if isinstance(start, pd.Series) and isinstance(end, pd.Series) \
      and len(intervals) > len(start):
        raise IndexError(
            "Indices of `start` and `end` are assumed to be aligned if present. "
            "If they are not aligned (i.e. sets of their values differ) then the result is wrong!"
        )
    mask_ok = intervals > '0 days'
    idx_ok = intervals.index[mask_ok]

    name = add_start if isinstance(add_start, str) else 'date_start'
    start = pd.DataFrame({'_': intervals, name: start})[name]  # only to properly 'elongate' start

    np.random.seed(seed)

    if any(mask_ok):
        deltas = np.random.randint([0] * n_samples, intervals.loc[idx_ok].dt.days.values[:, np.newaxis])
        if not isinstance(start, pd.Series):
            start_ok = pd.Series(start[mask_ok], index=idx_ok)
        else:
            start_ok = start.loc[idx_ok]
        res = pd.DataFrame(start_ok.to_numpy()[:, np.newaxis] + deltas.astype('<m8[D]'), index=idx_ok)
    else:
        res = pd.DataFrame(columns=range(n_samples))

    if any(~mask_ok):
        idx_bad = intervals.index[~mask_ok]
        res_bad = pd.DataFrame(index=idx_bad)
        res = pd.concat([res, res_bad]).loc[intervals.index]
        res = res.replace(np.nan, pd.NaT)   # type: ignore  # [call-overload]   so why it works?...

    if add_start:
        res = pd.concat([start, res], axis=1)
    if add_end:
        name = add_end if isinstance(add_end, str) else 'date_end'
        end = pd.DataFrame({'_': intervals, name: end})[name]  # only to properly 'elongate' end
        res = pd.concat([res, end], axis=1)

    if as_lists:
        res = res.apply(list, axis=1)   # type: ignore  # redefinition not allowed

    return res
