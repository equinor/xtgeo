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


def test_blockedwell_gridname_property(loadwell1):
    """Test gridname property getter and setter."""
    mywell = loadwell1

    # Test initial value (should be None)
    assert mywell.gridname is None

    # Test setting a valid gridname
    mywell.gridname = "MyGrid"
    assert mywell.gridname == "MyGrid"

    # Test setting another gridname
    mywell.gridname = "AnotherGrid"
    assert mywell.gridname == "AnotherGrid"

    # Test that invalid input raises ValueError
    with pytest.raises(ValueError, match="Input name is not a string"):
        mywell.gridname = 123

    with pytest.raises(ValueError, match="Input name is not a string"):
        mywell.gridname = ["list"]

    with pytest.raises(ValueError, match="Input name is not a string"):
        mywell.gridname = None


def test_blockedwell_copy(loadwell1):
    """Test that copy() method works and preserves gridname."""
    mywell = loadwell1

    # Set a gridname
    mywell.gridname = "TestGrid"

    # Copy the well
    copied_well = mywell.copy()

    # Verify the copy has the same gridname
    assert copied_well.gridname == "TestGrid"

    # Verify they are independent
    copied_well.gridname = "DifferentGrid"
    assert mywell.gridname == "TestGrid"
    assert copied_well.gridname == "DifferentGrid"

    # Verify other attributes are copied
    assert copied_well.wellname == mywell.wellname
    assert copied_well.xpos == mywell.xpos
    assert copied_well.ypos == mywell.ypos


def test_blockedwell_init_sets_gridname_to_none():
    """Test that __init__ sets gridname to None by default."""
    import pandas as pd

    # Create a BlockedWell directly
    bw = xtgeo.BlockedWell(
        rkb=100.0,
        xpos=1000.0,
        ypos=2000.0,
        wname="TestWell",
        df=pd.DataFrame(
            {
                "X_UTME": [1000.0, 1001.0],
                "Y_UTMN": [2000.0, 2001.0],
                "Z_TVDSS": [3000.0, 3001.0],
            }
        ),
    )

    # Verify gridname is initialized to None
    assert bw.gridname is None

    # Verify it's a BlockedWell instance
    assert isinstance(bw, xtgeo.BlockedWell)
    assert isinstance(bw, xtgeo.Well)
