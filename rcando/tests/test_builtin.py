#! python3
# -*- coding: utf-8 -*-
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
    name: test_builtin.py
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
from builtin import *

#%%
class TestLengthen(unittest.TestCase):

    def test_01(self):
        assert lengthen([1, 2], 4) == [1, 2, 2, 2]

    def test_02(self):
        assert lengthen([1, 2, 3, 4], 2) == [1, 2]

    def test_03(self):
        assert lengthen([[1, 2], [3, 4]], 4) == [[1, 2], [3, 4], [3, 4], [3, 4]]

    def test_04(self):
        assert lengthen(([1, 2], [3, 4]), 4) == ([1, 2], [3, 4], [3, 4], [3, 4])

    def test_05(self):
        assert lengthen(((1, 2), (3, 4)), 4) == ((1, 2), (3, 4), (3, 4), (3, 4))

    def test_06(self):
        assert lengthen([list('abc'), range(3)], 3) == [list('abc'), range(3), range(3)]



#%%
class TestFlatten(unittest.TestCase):

    def test_type(self):
        with self.assertRaises(TypeError):
            flatten(None)

    def test_str(self):
        assert flatten('str') == list('str')

    def test_00(self):
        self.assertEqual(flatten([]), [], "flatten([]) != []")

    def test_01(self):
        self.assertEqual(flatten([1, [2, [3]]]), [1, 2, 3], "flatten([1, [2, [3]]]) != [1, 2, 3]")

    def test_02(self):
        assert flatten([1, 2, [[3, 4, [5]], 6, 7], 8]) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_03(self):
        assert flatten([1, 2, [["3, 4", [5]], 6, 7], "8"]) == [1, 2, "3, 4", 5, 6, 7, "8"]

    def test_04(self):
        assert flatten([1, 2, [[(3, 4), [5]], {6, 7}], "8"]) == [1, 2, (3, 4), 5, {6, 7}, "8"]


#%%
class TestPaste(unittest.TestCase):

    def test_01(self):
        self.assertEqual(paste(['a', 'b'], [1, 2]),
                         ['a1', 'a2', 'b1', 'b2'], "!")

    def test_02(self):
        self.assertEqual(paste(['a', 'b'], [1, 2], False),
                         ['a1', 'b1', 'a2', 'b2'], "!")

    def test_03(self):
        self.assertEqual(paste(['a', 'b'], [1, 2], sep="_"),
                         ['a_1', 'a_2', 'b_1', 'b_2'], "!")

    def test_04(self):
        # the same as above
        self.assertEqual(paste(['a', 'b'], [1, 2], sep="_"),
                         ['a_1', 'a_2', 'b_1', 'b_2'], "!")

    def test_05(self):
       self.assertEqual(paste('str', range(3)),
                        ['str0', 'str1', 'str2'], "!")

    def test_06(self):
        self.assertEqual(paste(range(2), range(4,6), sep='_'),
                         ['0_4', '0_5', '1_4', '1_5'], "!")

    def test_07(self):
        self.assertEqual(paste(range(2), range(4,6), False, sep='_'),
                         ['0_4', '1_4', '0_5', '1_5'], "!")

    def test_main_case(self):
        self.assertEqual(paste(list('ABC'), range(3)),
                         ['A0', 'A1', 'A2', 'B0', 'B1', 'B2', 'C0', 'C1', 'C2'], "!")

    def test_no_flat(self):
        self.assertEqual(paste(list('ab'), range(2), flat=False),
                         [['a0', 'a1'], ['b0', 'b1']], "!")

    def test_left_simple(self):
        self.assertEqual(paste('a', range(3)),
                         ['a0', 'a1', 'a2'], "!")

    def test_right_simple(self):
        self.assertEqual(paste(list('abc'), 0),
                         ['a0', 'b0', 'c0'], "!")

    def test_both_simple(self):
        self.assertEqual(paste('a', 0), 'a0', "!")

#%%
if __name__ == "__main__":
    unittest.main()

