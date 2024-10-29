"""
Make a regularmap from a property at a given K layer
By JRIV
"""

import os

import xtgeo

GNAMEROOT = "../../xtgeo-testdata/3dgrids/reek/REEK"


def make_map():
    """Make a map of poro or perm in lowermost K layer of the grid"""

    # read grid
    grd = xtgeo.grid_from_file(GNAMEROOT + ".EGRID")

    poro = xtgeo.gridproperty_from_file(GNAMEROOT + ".INIT", name="PORO", grid=grd)

    poro_surface = xtgeo.surface_from_grid3d(grd, property=poro, where="base")

    # optional plot
    if "SKIP_PLOT" not in os.environ:
        poro_surface.quickplot()


if __name__ == "__main__":
    make_map()
