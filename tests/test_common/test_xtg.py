# -*- coding: utf-8 -*-
import time

import pytest

from xtgeo.common import timeit


def test_timer():
    """Test the timer function"""
    with timeit() as elapsed:
        assert pytest.approx(0, abs=0.001) == elapsed().total_seconds()
        time.sleep(0.001)
        assert pytest.approx(0.001, abs=0.001) == elapsed().total_seconds()

    # After ctx. exit time stops.
    assert pytest.approx(0.001, abs=0.001) == elapsed().total_seconds()
    time.sleep(0.001)
    assert pytest.approx(0.001, abs=0.001) == elapsed().total_seconds()
