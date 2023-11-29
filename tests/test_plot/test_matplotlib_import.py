import os
import sys
from unittest import mock


def _clear_state(sys, os):
    delete = []
    for module, _ in sys.modules.items():
        if module.startswith(("xtgeo", "matplotlib")):
            delete.append(module)

    for module in delete:
        del sys.modules[module]

    if "MPLBACKEND" in os.environ:
        del os.environ["MPLBACKEND"]


@mock.patch.dict(sys.modules)
@mock.patch.dict(os.environ)
def test_that_mpl_dynamically_imports():
    _clear_state(sys, os)
    import xtgeo  # noqa

    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules

    from xtgeo.plot.baseplot import BasePlot

    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules

    baseplot = BasePlot()

    assert "matplotlib" in sys.modules
    assert "matplotlib.pyplot" not in sys.modules

    baseplot.close()

    assert "matplotlib.pyplot" in sys.modules


@mock.patch.dict(sys.modules)
@mock.patch.dict(os.environ, {"LSB_JOBID": "1"})
def test_that_agg_backend_set_when_lsf_job():
    _clear_state(sys, os)
    import xtgeo  # noqa

    try:
        import roxar  # noqa

        assert os.environ.get("MPLBACKEND", "") == ""
    except ImportError:
        assert os.environ.get("MPLBACKEND", "") == "Agg"


@mock.patch.dict(sys.modules)
@mock.patch.dict(os.environ, {"DISPLAY": "X"})
def test_that_agg_backend_set_when_display_set():
    _clear_state(sys, os)
    import xtgeo  # noqa

    assert os.environ.get("MPLBACKEND", "") == ""
