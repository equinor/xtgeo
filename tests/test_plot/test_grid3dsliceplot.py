import pathlib

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.plot import Grid3DSlice

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

USEFILE1 = pathlib.Path("3dgrids/reek/reek_sim_grid.roff")
USEFILE2 = pathlib.Path("3dgrids/reek/reek_sim_poro.roff")
USEFILE3 = pathlib.Path("etc/colortables/rainbow_reverse.rmscolor")


@pytest.mark.bigtest
def test_slice_simple_layer(tmp_path, show_plot, generate_plot, testdata_path):
    """Trigger XSection class, and do some simple things basically."""
    layslice = Grid3DSlice()

    mygrid = xtgeo.grid_from_file(testdata_path / USEFILE1)
    myprop = xtgeo.gridproperty_from_file(
        testdata_path / USEFILE2, grid=mygrid, name="PORO"
    )

    assert myprop.values.mean() == pytest.approx(0.1677, abs=0.001)

    wd = None  # [457000, 464000, 1650, 1800]
    for lay in range(1, mygrid.nlay + 1):
        layslice.canvas(title=f"My Grid Layer plot for layer {lay}")
        layslice.plot_gridslice(
            mygrid,
            prop=myprop,
            mode="layer",
            index=lay,
            window=wd,
            linecolor="black",
        )

        if show_plot:
            layslice.show()
        if generate_plot:
            layslice.savefig(tmp_path / f"layerslice_{lay}.png")
        layslice.close()
