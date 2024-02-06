import os
import sys
from unittest import mock

import pytest
from packaging.version import parse as versionparse


def _clear_state(sys, os):
    delete = []
    for module, _ in sys.modules.items():
        if module.startswith(("xtgeo", "matplotlib")):
            delete.append(module)

    for module in delete:
        del sys.modules[module]

    if "MPLBACKEND" in os.environ:
        del os.environ["MPLBACKEND"]


@pytest.mark.skipif("ROXENV" in os.environ, reason="Dismiss test in ROXENV")
@mock.patch.dict(sys.modules)
@mock.patch.dict(os.environ)
def test_that_mpl_dynamically_imports():
    _clear_state(sys, os)
    import xtgeo  # noqa  # type:ignore

    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules

    from xtgeo.plot.baseplot import BasePlot

    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules

    baseplot = BasePlot()

    assert "matplotlib" in sys.modules

    import matplotlib as mpl

    if versionparse(mpl.__version__) < versionparse("3.6"):
        assert "matplotlib.pyplot" in sys.modules
    else:
        assert "matplotlib.pyplot" not in sys.modules

    baseplot.close()

    assert "matplotlib.pyplot" in sys.modules


@mock.patch("platform.system", return_value="Linux")
@mock.patch.dict(sys.modules)
@mock.patch.dict(os.environ, {"LSB_JOBID": "1"})
def test_that_agg_backend_set_when_lsf_job(mock_system):
    _clear_state(sys, os)
    import xtgeo  # noqa

    mock_system.assert_called_once()

    try:
        import roxar  # noqa

        assert os.environ.get("MPLBACKEND", "") == ""
    except ImportError:
        assert os.environ.get("MPLBACKEND", "") == "Agg"


@mock.patch("platform.system", return_value="Windows")
@mock.patch.dict(sys.modules)
def test_that_agg_backend_not_set_windows(mock_system):
    _clear_state(sys, os)
    import xtgeo  # noqa

    mock_system.assert_called_once()

    assert os.environ.get("MPLBACKEND", "") == ""


@mock.patch("platform.system", return_value="Darwin")
@mock.patch.dict(sys.modules)
def test_that_agg_backend_not_set_darwin(mock_system):
    _clear_state(sys, os)
    import xtgeo  # noqa

    mock_system.assert_called_once()

    assert os.environ.get("MPLBACKEND", "") == ""


@mock.patch.dict(sys.modules)
@mock.patch.dict(os.environ, {"DISPLAY": "X"})
def test_that_agg_backend_set_when_display_set():
    _clear_state(sys, os)
    import xtgeo  # noqa

    assert os.environ.get("MPLBACKEND", "") == ""
