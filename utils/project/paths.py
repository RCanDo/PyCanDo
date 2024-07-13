#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
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
from utils.builtin import Repr


# %%
class Paths(Repr):
    """
    simple proper & standard paths 'generator';
    common structure across projects,
    however still a bit customisable;
    """

    def __init__(
        self, data_version=None,
        root=None,  # root folder of a project; do not tinker with it! (if you don't have to)
        models="models",
        data="data",
        raw="raw",
        prep="prep",
        app="app",
        split="split",
        # config="configs"  # config shall be in  common.config.... files
        models_extra_level=None,
        pipe=None,  # alias for `models_extra_levels`; if not None then overwrites it
    ):
        """"""
        if root is None:
            pth = Path(__file__).absolute()  # .parents[2].resolve()
            # ROOT directory of a project is where .git folder resides, when used from within
            # regular project tree (having git repo established). Additional conditions prevent
            # from going into an infinite loop. This situation might happen e.g., when we are
            # building a docker image, where we put everything in the /home/user folder.
            while not (pth / ".git").exists() and not (pth / ".root_is_here").exists() and pth != Path("/"):
                pth = pth.parent
            self.ROOT = pth
        else:
            self.ROOT = Path(root)

        self.APP = self.ROOT / app
        # self.CONFIG = self.ROOT / config

        self.DATA = self.ROOT / data
        self.MODELS = self.ROOT / models

        self.MODELS /= pipe or models_extra_level or ""

        self.ROOT_DATA = self.DATA          # if `data_version` is not None then it's good to remember root values
        self.ROOT_MODELS = self.MODELS      # "

        if data_version is not None:  # usually it's NOT None
            self.DATA = self.DATA / data_version
            self.MODELS = self.MODELS / data_version

        self.DATA_RAW = self.DATA / raw
        self.DATA_CSV = self.DATA_RAW   # alias

        self.DATA_PREP = self.DATA / prep
        self.DATA_PKL = self.DATA_PREP  # alias

        self.DATA_SPLIT = self.DATA / split

        # # useable only if one leading data version
        # self.DATA_CURRENT = self.DATA_PREP / "data.pkl"    # .csv preprocessed to .pkl (usualy only proper types)

# %%
