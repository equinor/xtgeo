"""
Slice a Cube with a surface, and get attributes between two horizons

JRIV
"""

import pathlib
import tempfile

import xtgeo

DEBUG = False

EXPATH1 = pathlib.Path("../../xtgeo-testdata/cubes/etc/")
EXPATH2 = pathlib.Path("../../xtgeo-testdata/surfaces/etc")

TMPDIR = pathlib.Path(tempfile.gettempdir())


def slice_a_cube_with_surface():
    """Slice a seismic cube with a surface on OW dat/map format"""

    cubefile = EXPATH1 / "ib_test_cube2.segy"
    surfacefile = EXPATH2 / "h1.dat"

    mycube = xtgeo.cube_from_file(cubefile)

    # import map/dat surface using cube as template (inline/xline
    # must match)
    mysurf = xtgeo.surface_from_file(surfacefile, fformat="ijxyz", template=mycube)

    # sample cube values to mysurf (replacing current depth values)
    mysurf.slice_cube(mycube, sampling="trilinear")

    # export result
    mysurf.to_file(TMPDIR / "slice.dat", fformat="ijxyz")


def attribute_around_surface_symmetric():
    """Get attribute around a surface (symmetric)"""

    cubefile = EXPATH1 / "ib_test_cube2.segy"
    surfacefile = EXPATH2 / "h1.dat"

    mycube = xtgeo.cube_from_file(cubefile)

    mysurf = xtgeo.surface_from_file(surfacefile, fformat="ijxyz", template=mycube)

    attrs = ["max", "mean"]

    myattrs = mycube.compute_attributes_in_window(mysurf - 10.0, mysurf + 10.0)

    for attr in attrs:
        myattrs[attr].to_file(
            TMPDIR / ("myfile_symmetric_" + attr + ".dat"), fformat="ijxyz"
        )


def attribute_around_surface_asymmetric():
    """Get attribute around a surface (asymmetric)"""

    cubefile = EXPATH1 / "ib_test_cube2.segy"
    surfacefile = EXPATH2 / "h1.dat"

    above = 10
    below = 20

    mycube = xtgeo.cube_from_file(cubefile)

    mysurf = xtgeo.surface_from_file(surfacefile, fformat="ijxyz", template=mycube)

    myattrs = mycube.compute_attributes_in_window(mysurf - above, mysurf + below)

    for attr in myattrs:
        if DEBUG:
            myattrs[attr].describe()

        myattrs[attr].to_file(
            TMPDIR / ("myfile_asymmetric_" + attr + ".dat"), fformat="ijxyz"
        )


def attribute_around_constant_cube_slices():
    """Get attribute around a constant cube slices"""

    cubefile = EXPATH1 / "ib_test_cube2.segy"

    level1 = 1010.0
    level2 = 1100.0

    mycube = xtgeo.cube_from_file(cubefile)

    myattrs = mycube.compute_attributes_in_window(level1, level2)

    for attr in myattrs:
        if DEBUG:
            myattrs[attr].describe()

        myattrs[attr].to_file(
            TMPDIR / ("myfile_constlevels_" + attr + ".dat"), fformat="ijxyz"
        )


if __name__ == "__main__":
    slice_a_cube_with_surface()
    attribute_around_surface_symmetric()
    attribute_around_surface_asymmetric()
    attribute_around_constant_cube_slices()

    print(f"Running example OK: {pathlib.Path(__file__).name}")
