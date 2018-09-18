# coding=utf-8
from __future__ import print_function, division, unicode_literals
from psy import SimAdaptiveTirt


def test_cat():
    sat = SimAdaptiveTirt(subject_nums=1, item_size=600, trait_size=30, max_sec_item_size=40)
    sat.sim()