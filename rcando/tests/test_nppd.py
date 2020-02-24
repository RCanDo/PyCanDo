# -*- coding: utf-8 -*-
#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

type: tests
version: 1.0
todo:
file:
    usage:
        interactive: True
        terminal: True
    name: test_nppd.py
    path: D:/ROBOCZY/Python/RCanDo/ak/
    date: 2019-11-20
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

#%%
"""
pwd
cd D:/ROBOCZY/Python/RCanDo/...
ls
"""
import unittest
from nppd import data_frame

import numpy as np
import pandas as pd

#%%

class TestDataFrame(unittest.TestCase):

    def test_01(self):
        df_test = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                                'B': ['B0', 'B1', 'B2', 'B3'],
                                'C': ['C0', 'C1', 'C2', 'C3'],
                                'D': ['D0', 'D1', 'D2', 'D3']})
        df = data_frame(list('ABCD'), range(4))
        self.assertTrue(df.equals(df_test), "!")

    def test_02(self):
        df_test = pd.DataFrame({'A': ['A2', 'A3', 'A4', 'A5'],
                                'B': ['B2', 'B3', 'B4', 'B5']})
        df = data_frame(list('AB'), range(2, 6))
        self.assertTrue(df.equals(df_test), "!")

    def test_03(self):
        df_test = pd.DataFrame({'A': ['A2', 'A3', 'A4', 'A5'],
                                'B': ['B2', 'B3', 'B4', 'B5']},
                               index=range(2, 6) )
        df = data_frame(list('AB'), range(2, 6), index=True)
        self.assertTrue(df.equals(df_test), "!")

    def test_04(self):
        df_test = pd.DataFrame({'A': ['A2', 'A3', 'A4', 'A5'],
                                'B': ['B2', 'B3', 'B4', 'B5']},
                               index=list('pqrs') )
        df = data_frame(list('AB'), range(2, 6), index=list('pqrs'))
        self.assertTrue(df.equals(df_test), "!")

    def test_05(self):
        df_test = pd.DataFrame({'X': ['A2', 'A3', 'A4', 'A5'],
                                'Y': ['B2', 'B3', 'B4', 'B5']},
                               index=list('pqrs'))
        df = data_frame(list('AB'), range(2, 6), index=list('pqrs'), columns=list('XY'))
        self.assertTrue(df.equals(df_test), "!")

    def test_06(self):
        df_test = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                                'B': ['B0', 'B0', 'B2', 'B2'],
                                'C': ['C1', 'C3', 'C5', 'C7']})
        df = data_frame(list('ABC'), [range(4), list('0022'), [1, 3, 5, 7]])
        self.assertTrue(df.equals(df_test), "!")

    def test_07(self):
        df_test = pd.DataFrame({'X': ['A1', 'A2', 'A3', 'A4'],
                                'Y': ['B0', 'B0', 'B2', 'B2'],
                                'Z': ['C1', 'C3', 'C5', 'C7']})
        df = data_frame(list('ABC'), [range(1, 5), list('0022'), [1, 3, 5, 7]],
                        columns=list('XYZ'))
        self.assertTrue(df.equals(df_test), "!")

    def test_08(self):
        df_test = pd.DataFrame({'X': ['A1', 'A2', 'A3', 'A4'],
                                'Y': ['B0', 'B0', 'B2', 'B2'],
                                'Z': ['C1', 'C3', 'C5', 'C7']},
                                index=[1, 2, 3, 4])
        df = data_frame(list('ABC'), [range(1, 5), list('0022'), [1, 3, 5, 7]],
                        columns=list('XYZ'), index=True)
        self.assertTrue(df.equals(df_test), "!")



#%%

if __name__ == "__main__":
    unittest.main()