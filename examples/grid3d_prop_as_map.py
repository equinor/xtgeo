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

    _ = xtgeo.gridproperty_from_file(GNAMEROOT + ".INIT", name="PORO", grid=grd)

    df = grd.get_dataframe()

    # make a map from the grid geometry to be used as a template

    surf = xtgeo.surface_from_grid3d(grd)

    # get only bottom layer:
    lastlayer = df["KZ"].max()
    df = df[df["KZ"] == lastlayer].reset_index()

    # prepare as input to a Points dataframe (3 columns X Y Z)
    df = df[["X_UTME", "Y_UTMN", "PORO"]].copy()

    points = xtgeo.Points()
    points.zname = "PORO"
    points.set_dataframe(df)

    # do gridding:
    surf.gridding(points)

    # optional plot
    if "SKIP_PLOT" not in os.environ:
        surf.quickplot()


if __name__ == "__main__":
    make_map()
