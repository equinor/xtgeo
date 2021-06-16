"""
Slice a 3Grid property with a surface, e.g. a FLW map.

In this case 3 maps with constant depth are applied. The maps are refined
for smoother result, and output is exported as Roxar binary *.gri and
quickplots (png)

JRIV
"""
import os
import pathlib
import tempfile

import xtgeo

TMPDIR = pathlib.Path(tempfile.gettempdir())


def slice_a_grid():
    """Slice a 3D grid property with maps (looping)"""

    expath1 = pathlib.Path("../../xtgeo-testdata/3dgrids/reek")
    expath2 = pathlib.Path("../../xtgeo-testdata/surfaces/reek/1")

    gridfileroot = expath1 / "REEK"
    surfacefile = expath2 / "midreek_rota.gri"

    initprops = ["PORO", "PERMX"]

    grd = xtgeo.grid_from_file(gridfileroot, fformat="eclipserun", initprops=initprops)

    # read a surface, which is used for "template"
    surf = xtgeo.surface_from_file(surfacefile)
    surf.refine(2)  # make finer for nicer sampling (NB takes time then)

    slices = [1700, 1720, 1740]

    for myslice in slices:

        print("Slice is {}".format(myslice))

        for prp in grd.props:
            sconst = surf.copy()
            sconst.values = myslice  # set constant value for surface

            print("Work with {}, slice at {}".format(prp.name, myslice))
            sconst.slice_grid3d(grd, prp)

            fname = "{}_{}.gri".format(prp.name, myslice)
            sconst.to_file(TMPDIR / fname)

            fname = TMPDIR / ("{}_{}.png".format(prp.name, myslice))

            if "SKIP_PLOT" not in os.environ:
                sconst.quickplot(filename=fname)


if __name__ == "__main__":
    slice_a_grid()

    print(f"Running example OK: {pathlib.Path(__file__).name}")
