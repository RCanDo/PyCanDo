#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title:
subtitle:
version: 1.0
type: module
keywords: [kw1, ...]   # there are always some keywords!
description: |
remarks:
todo:
    - How to use it as module which loads  `black.mplstyle`  from its directory ???
sources:
    - title:
      chapter:
      pages:
      link:
      date:
      authors:
          - nick:
            fullname:
            email:
      usage: |
          idea & inspiration
file:
    usage:
        interactive: True
        terminal: True
    name: _.py
    path: ~/Projects/Python/RCanDo/..
    date: 2021-09-27
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""

#%%
"""
pwd
cd ~/Projects/Python/RCanDo/...
ls
"""


#%%
def set_style(style='', style_default=path_wd+'/black.mplstyle', c='', c_default='y'):
    '''
    c : color;

    style : mpl.style
    '''
    global path_wd

    if c=='': c=c_default
    if style=='': style=style_default  ##'dark_background'

    if style in [ 'dark_background', path_wd+'/black.mplstyle' ]:
        c = c.replace('b','y').replace('k','w')

    plt.style.use(style)

    return style, c

#%%