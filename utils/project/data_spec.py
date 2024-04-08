#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: project utilities
version: 1.0
type: module
keywords: [project, parameters, documents, ...]
description: |
    Data management helpers;
    mainly data structures for storing all kind of parameters.
content:
remarks:
todo:
sources:
file:
    date: 2022-12-27
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
import utils.builtin as bi
# from attrdict import AttrDict
from dataclasses import dataclass, field  # fields


# %%
# abandoned (unnecessary complication)
@bi.repr
@dataclass
class Files:
    RAW: dict = field(default_factory=lambda: dict())
    PREP: dict = field(default_factory=lambda: dict())


@bi.repr
@dataclass
class DataSpec:
    # some description of data
    description: str = \
        """raw data;
        """  # E122
    # FILE: Files = Files()   # abandoned (unnecessary complication)

# %%
