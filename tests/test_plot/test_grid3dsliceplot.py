# -*- coding: utf-8 -*-


import os

import pytest

from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid, GridProperty
from xtgeo.plot import Grid3DSlice

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

if "XTG_SHOW" in os.environ:
    XTGSHOW = True
else:
    XTGSHOW = False

logger.info("Use env variable XTG_SHOW to show interactive plots to screen")

TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

USEFILE1 = TPATH / "3dgrids/reek/reek_sim_grid.roff"
USEFILE2 = TPATH / "3dgrids/reek/reek_sim_poro.roff"
USEFILE3 = TPATH / "etc/colortables/rainbow_reverse.rmscolor"


@pytest.mark.skipifroxar
def test_slice_simple_layer(tmpdir):
    """Trigger XSection class, and do some simple things basically."""
    layslice = Grid3DSlice()

    mygrid = Grid(USEFILE1)
    myprop = GridProperty(USEFILE2, grid=mygrid, name="PORO")

    assert myprop.values.mean() == pytest.approx(0.1677, abs=0.001)

    wd = None  # [457000, 464000, 1650, 1800]
    for lay in range(1, mygrid.nlay + 1):
        layslice.canvas(title="My Grid Layer plot for layer {}".format(lay))
        layslice.plot_gridslice(
            mygrid,
            prop=myprop,
            mode="layer",
            index=lay,
            window=wd,
            linecolor="black",
        )

        if XTGSHOW:
            layslice.show()
        else:
            print(
                "Output to screen disabled (will plot to screen); "
                "use XTG_SHOW env variable"
            )
            layslice.savefig(os.path.join(tmpdir, "layerslice_" + str(lay) + ".png"))
