"""
Print a CSV from all INIT vectors
"""

import pathlib
import tempfile
import numpy as np
import xtgeo

EXPATH = pathlib.Path("../../xtgeo-testdata/3dgrids/reek")

GRIDFILEROOT = EXPATH / "REEK"
TMPDIR = pathlib.Path(tempfile.gettempdir())

INITPROPS = "all"  # will look for all vectors that looks "gridvalid"


def all_init_as_csv():
    """Get dataframes, print as CSV."""

    print("Loading Eclipse data {}".format(GRIDFILEROOT))
    grd = xtgeo.grid_from_file(GRIDFILEROOT, fformat="eclipserun", initprops=INITPROPS)
    print("Get dataframes...")
    dfr = grd.get_dataframe(activeonly=True)

    print(dfr.head())
    print("Filter out columns with constant values...")
    dfr = dfr.iloc[:, ~np.isclose(0, dfr.var())]
    print(dfr.head())
    print("Write to file...")
    dfr.to_csv(TMPDIR / "mycsvdump.csv", index=False)


if __name__ == "__main__":

    all_init_as_csv()
