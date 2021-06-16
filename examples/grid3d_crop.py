"""
Crop a 3D grid.
"""

import tempfile
import pathlib

import xtgeo

EXPATH1 = pathlib.Path("../../xtgeo-testdata/3dgrids/reek")

GRIDFILEROOT = EXPATH1 / "REEK"

INITPROPS = ["PORO", "PERMX"]

TMPDIR = pathlib.Path(tempfile.gettempdir())


def cropper():
    """Do a cropping of a 3D grid"""

    # pylint: disable=too-many-locals
    grd = xtgeo.grid_from_file(GRIDFILEROOT, fformat="eclipserun", initprops=INITPROPS)

    print(grd.props)

    # find current NCOL, NROW and divide into 4 pieces

    ncol = grd.ncol
    nrow = grd.nrow
    nlay = grd.nlay

    ncol1 = int(ncol / 2)

    nrow1 = int(nrow / 2)

    print("Original grid dimensions are {} {} {}".format(ncol, nrow, nlay))
    print("Crop ranges are {} {} {}".format(ncol1, nrow1, nlay))

    ncolranges = [(1, ncol1), (ncol1 + 1, ncol)]
    nrowranges = [(1, nrow1), (nrow1 + 1, nrow)]

    for ncr in ncolranges:
        nc1, nc2 = ncr
        for nrr in nrowranges:

            nr1, nr2 = nrr

            fname = "_{}-{}_{}-{}".format(nc1, nc2, nr1, nr2)

            tmpgrd = grd.copy()
            tmpgrd.crop(ncr, nrr, (1, nlay), props="all")
            # save to disk as ROFF files
            tmpgrd.to_file(TMPDIR / ("grid" + fname + ".roff"))
            for prop in tmpgrd.props:
                print("{} for {} .. {}".format(prop.name, ncr, nrr))
                fname2 = prop.name + fname + ".roff"
                fname2 = fname2.lower()
                prop.to_file(TMPDIR / fname2)


if __name__ == "__main__":
    cropper()
