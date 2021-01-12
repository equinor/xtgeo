# -*- coding: utf-8 -*-


import os
import matplotlib.pyplot as plt

# import pytest

# from xtgeo.plot import Grid3DSlice
# from xtgeo.grid3d import Grid
# from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog
import tests.test_common.test_xtg as tsetup


xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

if "XTG_SHOW" in os.environ:
    XTGSHOW = True
else:
    XTGSHOW = False

logger.info("Use env variable XTG_SHOW to show interactive plots to screen")

TMPDIR = xtg.tmpdir
TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

USEFILE1 = TPATH / "3dgrids/reek/reek_sim_grid.roff"
USEFILE2 = TPATH / "3dgrids/reek/reek_sim_poro.roff"
USEFILE3 = TPATH / "etc/colortables/rainbow_reverse.rmscolor"


@tsetup.skipifroxar
def test_very_basic():
    """Just test that matplotlib works."""
    assert "matplotlib" in str(plt)

    plt.title("Hello world")
    plt.savefig(os.path.join(TMPDIR, "helloworld1.png"))
    plt.savefig(os.path.join(TMPDIR, "helloworld1.svg"))
    if XTGSHOW:
        plt.show()
    logger.info("Very basic plotting")
    plt.close()


# @tsetup.skipifroxar
# def test_slice_simple():
#     """Trigger XSection class, and do some simple things basically."""
#     layslice = Grid3DSlice()

#     mygrid = Grid(USEFILE1)
#     myprop = GridProperty(USEFILE2, grid=mygrid, name="PORO")

#     assert myprop.values.mean() == pytest.approx(0.1677, abs=0.001)

#     layslice.canvas(title="My Grid plot")
#     wnd = (454000, 455000, 6782000, 6783000)
#     layslice.plot_gridslice(mygrid, myprop, window=wnd, colormap=USEFILE3)

#     if XTGSHOW:
#         layslice.show()
#     else:
#         print(
#             "Output to screen disabled (will plotto screen); "
#             "use XTG_SHOW env variable"
#         )
#         layslice.savefig(os.path.join(TMPDIR, "layerslice.png"))


# @tsetup.skipifroxar
# def test_slice_plot_many_grid_layers():
#     """Loop over layers and produce both SVG and PNG files to file"""

#     mygrid = Grid(USEFILE1)
#     myprop = GridProperty(USEFILE2, grid=mygrid, name="PORO")

#     nlayers = mygrid.nlay + 1

#     layslice2 = Grid3DSlice()

#     for k in range(1, nlayers, 4):
#         print("Layer {} ...".format(k))
#         layslice2.canvas(title="Porosity for layer " + str(k))
#         layslice2.plot_gridslice(
#             mygrid, myprop, colormap=USEFILE3, index=k, minvalue=0.18, maxvalue=0.36
#         )
#         layslice2.savefig(
#             os.path.join(TMPDIR, "layerslice2_" + str(k) + ".svg"),
#             fformat="svg",
#             last=False,
#         )
#         layslice2.savefig(os.path.join(TMPDIR, "layerslice2_" + str(k) + ".png"))
