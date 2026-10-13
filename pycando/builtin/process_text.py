#! python3
# -*- coding: utf-8 -*-
"""
---
title: processing text
version: 1.0
type: module
keywords: [replace, regexp,  ]
description: |
    Convenience functions for text processing.
remarks:
    - We use only basic packages from standard library
      like functools, itertools, typing, time, math
todo:
sources:
file:
    date: 2024-12-06
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

import re


def process_text(
        text: str,
        replace: list[tuple[str, str]] | None = None,  # simpler then regexp
        #   (no need for escaping special characters like e.g. '.')
        resub: list[tuple[str, str]] | None = None,
        lower: bool = False,
        split: bool | str = False,
) -> str | list[str]:
    """
    Simple text processing: replacing via replace or regexp, lowering and splitting (in this order).
    All in one go, what allows to spare few lines in some situations
    e.g. when processing `pd.Series` entries via `.apply()`.
    Especially replacing/substituting number of cases is simplified as series of `(from_, to_)` directives
    may be conveniently passed in one list.

    Arguments
    ---------
    text: str,
        text (str) to be processed;
    replace: list[tuple[str, str]] = None,
        list of tuples `(from_, to_)` to be passed one by one to `text.replace(from_, to_)`;
        using `replace` option instead of `resub` is simpler as there is no need for escaping some characters,
        however not all, and no complicated patterns possible.
    resub: list[tuple[str, str]] = None,
        list of tuples `(from_, to_)` to be passed one by one to regular expression device which will
        replace pattern `from_` to string `to_`, e.g. `(r'^[a-z]', '$1_')`;
    lower: bool = False,
        lowering all the letters in text (after `replace` and `resub` done);
    split: bool | str = False,
        splitting `text` by the given sign;
        works the same as `text.split(split)` or `text.split()` if `split = False`;
        If `split` is not False then the result is list of strings.

    Examples
    --------
    text = "This is  trial text.   With few sentences.  There are  3of them. Or 4or more... O0!"
    process_text(text, resub = [(r'\b([0-9]+)', r'\1_'),])
    process_text(text, resub = [(r'\s+', ' '),])
    process_text(text, replace = [('  ', ' '),])
    process_text(text, resub = [('  ', ' '),])

    # using `replace` is simpler as there is generally no need for escaping interpunction characters,
    # however escape character `\` must be escaped even for `replace`
    process_text(r']q.q\-[ryqu?!', replace = [('\\', ''), ('.', ''), ('?', ''), ('!',''), ('[',''), (']',''),])

    text = "Oh k...i'm  watching   here:) "
    process_text(
        text,
        resub = [(r"[',\.\-\?\!0-9\(\)\"\/\:\£\$\>\<\*\;\\]+", " "),      # (repeated) interpunction to single space
                 (r"\s+", " "),        # repeated spaces to single space
                ],)
    """
    if replace:
        for from_, to_ in replace:
            text = text.replace(from_, to_)
    if resub:
        for regexp, sub in resub:
            reg = re.compile(regexp)
            text = reg.sub(sub, text)
    if lower:
        text = text.lower()
    if isinstance(split, bool):
        if split:
            text = text.split()
    elif isinstance(split, str):
        if split:
            text = text.split(split)
    return text
