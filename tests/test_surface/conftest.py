import numpy as np
import pytest


@pytest.fixture
def default_surface():
    yield {
        "ncol": 5,
        "nrow": 3,
        "xori": 0.0,
        "yori": 0.0,
        "xinc": 25.0,
        "yinc": 25.0,
        "values": np.array(
            [[1, 6, 11], [2, 7, 12], [3, 8, 1e33], [4, 9, 14], [5, 10, 15]],
            dtype=np.float64,
            order="C",
        ),
    }


@pytest.fixture
def larger_surface():
    yield {
        "ncol": 6,
        "nrow": 7,
        "xori": 0.0,
        "yori": 0.0,
        "xinc": 25.0,
        "yinc": 25.0,
        "values": np.array(
            [
                [0.0001, 6.0, 11.0, 88888.822, 1.001, 41.33, 1e33],
                [0.0002, 7.0, 1e33, 88888.822, 1.002, 42.33, 0.0],
                [0.0003, 8.0, 11.0, 88888.822, 1.003, 43.33, 1e33],
                [0.0004, 9.0, 1e33, 88888.822, 1.004, 44.33, 0.0],
                [0.0005, 10.0, 11.0, 88888.822, 1.005, 45.33, 0.0],
                [0.0006, 11.0, 11.0, 88888.822, 1.006, 46.33, 99.0],
            ],
            dtype=np.float64,
            order="C",
        ),
    }
