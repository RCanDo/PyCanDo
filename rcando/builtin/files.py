#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: files utilities
version: 1.0
type: module             # module, analysis, model, tutorial, help, example, ...
keywords: [file, read, open, context, ...]
description: |
sources:
remarks:
todo:
file:
    usage:
        interactive: False  # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    date: 2023-01-19
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""


# %%
def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def file_nrows_0(file_name: str) -> int:
    """
    number of rows in a text file;
    remember that for .csv file it includes header (column names),
    hence there are 1 less proper data rows (if header exists, what is standard)
    """
    with open(file_name, 'rb') as fp:
        c_generator = _count_generator(fp.read)
        # count each \n
        count = sum(buffer.count(b'\n') for buffer in c_generator)
    return count


# %%
def file_nrows(file_name: str) -> int:
    """
    bit faster then `file_nrows_0()`;
    number of rows in a text file;
    remember that for .csv file it includes header (column names),
    hence there are 1 less proper data rows (if header exists, what is standard)
    """
    with open(file_name, 'r') as fp:
        for count, line in enumerate(fp):
            pass
    return count + 1


# %%
""" testing
file_nrows_0('data/iter4/kampania_ustalania_ceny/raw/inventory_fixed.csv')      # 296149
file_nrows('data/iter4/kampania_ustalania_ceny/raw/inventory_fixed.csv')        # 296148
%timeit file_nrows_0('data/iter4/kampania_ustalania_ceny/raw/inventory_fixed.csv')
# 3.06 s ± 16.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 3.09 s ± 58.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 3.12 s ± 34.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit file_nrows('data/iter4/kampania_ustalania_ceny/raw/inventory_fixed.csv')
# 2.78 s ± 57.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 2.85 s ± 144 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 3.22 s ± 248 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""

# %%
