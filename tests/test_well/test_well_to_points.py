import pytest

from xtgeo.well import Well
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

TMPD = xtg.tmpdir

logger = xtg.basiclogger(__name__)

WFILE = "../xtgeo-testdata/wells/etc/otest.rmswell"


def test_wellzone_to_points():
    """Import well from file and put zone boundaries to a Pandas object."""

    mywell = Well(WFILE, zonelogname="Zone_model2", mdlogname="M_MDEPTH")

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=False)
    assert zpoints.iat[9, 6] == 6

    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=True)
    assert zpoints.iat[9, 6] == 7

    with pytest.raises(ValueError):
        zpoints = mywell.get_zonation_points(zonelist=[1, 3, 4, 5])

    zpoints = mywell.get_zonation_points(zonelist=[3, 4, 5])
    assert zpoints.iat[6, 6] == 4

    zpoints = mywell.get_zonation_points(zonelist=(3, 5))
    assert zpoints.iat[6, 6] == 4


def test_wellzone_to_isopoints():
    """Import well from file and find thicknesses"""

    mywell = Well(WFILE, zonelogname="Zone_model2", mdlogname="M_MDEPTH")
    # get the zpoints which is a Pandas
    zpoints = mywell.get_zonation_points(use_undef=False, tops=True)
    assert zpoints["Zone"].min() == 3
    assert zpoints["Zone"].max() == 9

    zisos = mywell.get_zonation_points(use_undef=False, tops=False)
    assert zisos.iat[10, 8] == 4
