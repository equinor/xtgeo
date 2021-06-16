"""
Example on how to retrieve a dataframe (Pandas) from a 3D grid.

Explanation:

Both a GridProperties and a Grid instance can return a dataframe.
The `grd.gridprops` attribute below is the GridProperties, and
this will return a a dataframe by default which does not include
XYZ and ACTNUM, as this information is only from the Grid (geometry).

The grid itself can also return a dataframe, and in this case
XYZ and ACNUM will be returned by default. Also properties that
are "attached" to the Grid via a GridProperties attribute will
be shown.

"""

import pathlib
import tempfile
import xtgeo

EXPATH = pathlib.Path("../../xtgeo-testdata/3dgrids/reek")

GRIDFILEROOT = EXPATH / "REEK"
TMPDIR = pathlib.Path(tempfile.gettempdir())

INITS = ["PORO", "PERMX"]
RESTARTS = ["PRESSURE", "SWAT", "SOIL"]
MYDATES = [20001101, 20030101]


def extractdf():
    """Extract dataframe from Eclipse case"""

    # gete dataframe from the grid only
    grd = xtgeo.grid_from_file(GRIDFILEROOT.with_suffix(".EGRID"))
    dataframe = grd.dataframe()  # will not have any grid props
    print(dataframe)

    # load as Eclipse run; this will automatically look for EGRID, INIT, UNRST
    grd = xtgeo.grid_from_file(
        GRIDFILEROOT,
        fformat="eclipserun",
        initprops=INITS,
        restartprops=RESTARTS,
        restartdates=MYDATES,
    )

    # dataframe from a GridProperties instance, in this case grd.gridprops
    dataframe = grd.gridprops.dataframe()  # properties for all cells

    print(dataframe)

    # Get a dataframe for all cells, with ijk and xyz. In this case
    # a grid key input is required:
    dataframe = grd.dataframe()

    print(dataframe)  # default is for all cells

    # For active cells only:
    dataframe = grd.dataframe(activeonly=True)

    print(dataframe)

    dataframe.to_csv(TMPDIR / "reek_sim.csv")


if __name__ == "__main__":
    extractdf()
