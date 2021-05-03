import numpy as np
import pytest


@pytest.fixture
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


@pytest.fixture
def default_surface():
    yield {
        "ncol": 5,
        "nrow": 3,
        "xori": 0.0,
        "yori": 0.0,
        "xinc": 25,
        "yinc": 25,
        "values": np.array(
            [[1, 6, 11], [2, 7, 12], [3, 8, 1e33], [4, 9, 14], [5, 10, 15]],
            dtype=np.float64,
            order="C",
        ),
    }
