#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Paths class
version: 1.0
type: module
keywords: [paths, directories, ...]
description: |
    Standard project directories
remarks:
todo:
sources:
file:
    date: 2022-10-11
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
from pathlib import Path
import utils.builtin as bi


# %%
class Paths(bi.Repr):
    """
    simple proper & standard paths 'generator';
    common structure across projects,
    however still bit customisable;
    """

    def __init__(
        self, data_version=None,
        root=None,  # root folder of a project; do not tinker with it! (if you don't have to)
        models="models",
        data="data",
        raw="raw",
        prep="prep",
        app="app",
        splitted="splitted"
        # config="configs"  # config shall be in  common.config.... files
    ):
        """"""
        if root is None:
            pth = Path(__file__).absolute()  # .parents[2].resolve()
            # ROOT directory of a project is where .git folder resides
            while not list(pth.glob(".git")):
                pth = pth.parent
            self.ROOT = pth
        else:
            self.ROOT = root

        self.APP = self.ROOT / app
        # self.CONFIG = self.ROOT / config

        self.DATA = self.ROOT / data
        self.MODELS = self.ROOT / models
        self.ROOT_DATA = self.DATA          # if `data_version` is not None then it's good to remember root values
        self.ROOT_MODELS = self.MODELS      # "

        if data_version is not None:        # usually it's NOT None
            self.DATA = self.DATA / data_version
            self.MODELS = self.MODELS / data_version

        self.DATA_RAW = self.DATA / raw
        self.DATA_CSV = self.DATA_RAW   # alias

        self.DATA_PREP = self.DATA / prep
        self.DATA_PKL = self.DATA_PREP  # alias

        self.DATA_SPLITTED = self.DATA / splitted
        self.DATA_SPLIT = self.DATA_SPLITTED    # alias

        # # useable only if one leading data version
        # self.DATA_CURRENT = self.DATA_PREP / "data.pkl"    # .csv preprocessed to .pkl (usualy only proper types)

# %%
