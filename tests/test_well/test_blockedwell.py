import logging
from os.path import join

import pytest

import xtgeo

logger = logging.getLogger(__name__)


@pytest.fixture(name="loadwell1")
def fixture_loadwell1(testdata_path):
    """Fixture for loading a well (pytest setup)"""
    logger.info("Load well 1")
    wfile = join(testdata_path, "wells/reek/1/OP_1.bw")
    return xtgeo.blockedwell_from_file(wfile)


def test_import_blockedwell(loadwell1):
    """Import blocked well from file."""

    mywell = loadwell1

    assert mywell.xpos == 461809.6, "XPOS"
    assert mywell.ypos == 5932990.4, "YPOS"
    assert mywell.wellname == "OP_1", "WNAME"
    assert mywell.xname == "X_UTME"

    assert mywell.get_logtype("Facies") == "DISC"
    assert mywell.get_logrecord("Facies") == {
        0: "Background",
        1: "Channel",
        2: "Crevasse",
    }

    assert mywell.get_dataframe()["Poro"][4] == pytest.approx(0.224485, abs=0.0001)


def test_blockedwell_hdf5_import_selected_logs(tmp_path, loadwell1):
    """Import blocked well from HDF5 with lognames filter and strict mode."""
    wname = (tmp_path / "blockedwell_log_filter").with_suffix(".hdf")
    loadwell1.to_file(wname, fformat="hdf5")

    result = xtgeo.blockedwell_from_file(wname, fformat="hdf5", lognames="Poro")
    assert "Poro" in result.get_dataframe()
    assert "Facies" not in result.get_dataframe()

    result = xtgeo.blockedwell_from_file(wname, fformat="hdf5", lognames=["DUMMY"])
    assert "Poro" not in result.get_dataframe()
    assert "Facies" not in result.get_dataframe()

    with pytest.raises(ValueError):
        xtgeo.blockedwell_from_file(
            wname,
            fformat="hdf5",
            lognames=["DUMMY"],
            lognames_strict=True,
        )
