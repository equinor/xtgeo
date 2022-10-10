# -*- coding: utf-8 -*-
"""Conftest functions"""
from distutils.version import LooseVersion

import pandas as pd
import pytest


class Helpers:
    @staticmethod
    def df2csv(dataframe, index=True):
        """Combat Pandas change 1.4 -> 1.5; avoid FutureWarning on line_terminator."""
        if LooseVersion(pd.__version__) < LooseVersion("1.5"):
            return dataframe.to_csv(line_terminator="\n", index=index)
        else:
            return dataframe.to_csv(lineterminator="\n", index=index)


@pytest.fixture
def helpers():
    return Helpers
