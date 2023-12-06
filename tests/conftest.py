"""Conftest functions"""
import pandas as pd
import pytest
from packaging.version import parse as versionparse


class Helpers:
    @staticmethod
    def df2csv(dataframe, index=True):
        """Combat Pandas change 1.4 -> 1.5; avoid FutureWarning on line_terminator."""
        if versionparse(pd.__version__) < versionparse("1.5"):
            return dataframe.to_csv(line_terminator="\n", index=index)
        return dataframe.to_csv(lineterminator="\n", index=index)


@pytest.fixture
def helpers():
    return Helpers
