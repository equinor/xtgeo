"""
Slice a Cube with a surface, and get attributes between two horizons

In this case 3 maps with constant depth are applied. The maps are refined
for smoother result, and output is exported as Roxar binary *.gri and
quickplots (png)

JRIV
"""

from os.path import join, basename

import xtgeo

DEBUG = False

EXPATH1 = "../../xtgeo-testdata/cubes/etc/"
EXPATH2 = "../../xtgeo-testdata/surfaces/etc"


def slice_a_cube_with_surface():
    """Slice a seismic cube with a surface on OW dat/map format"""

    cubefile = join(EXPATH1, "ib_test_cube2.segy")
    surfacefile = join(EXPATH2, "h1.dat")

    mycube = xtgeo.cube_from_file(cubefile)

    # import map/dat surface using cube as template (inline/xline
    # must match)
    mysurf = xtgeo.surface_from_file(surfacefile, fformat="ijxyz", template=mycube)

    # sample cube values to mysurf (replacing current depth values)
    mysurf.slice_cube(mycube, sampling="trilinear")

    # export result
    mysurf.to_file("slice.dat", fformat="ijxyz")


def attribute_around_surface_symmetric():
    """Get atttribute around a surface (symmetric)"""

    cubefile = join(EXPATH1, "ib_test_cube2.segy")
    surfacefile = join(EXPATH2, "h1.dat")

    mycube = xtgeo.cube_from_file(cubefile)

    mysurf = xtgeo.surface_from_file(surfacefile, fformat="ijxyz", template=mycube)

    attrs = ["max", "mean"]

    myattrs = mysurf.slice_cube_window(
        mycube, attribute=attrs, sampling="trilinear", zrange=10.0
    )
    for attr in myattrs.keys():
        myattrs[attr].to_file("myfile_symmetric_" + attr + ".dat", fformat="ijxyz")


def attribute_around_surface_asymmetric():
    """Get attribute around a surface (asymmetric)"""

    cubefile = join(EXPATH1, "ib_test_cube2.segy")
    surfacefile = join(EXPATH2, "h1.dat")

    above = 10
    below = 20

    mycube = xtgeo.cube_from_file(cubefile)

    mysurf = xtgeo.surface_from_file(surfacefile, fformat="ijxyz", template=mycube)

    # instead of using zrange, we make some tmp surfaces that
    # reflects the assymmetric
    sabove = mysurf.copy()
    sbelow = mysurf.copy()
    sabove.values -= above
    sbelow.values += below

    if DEBUG:
        sabove.describe()
        sbelow.describe()

    attrs = "all"

    myattrs = mysurf.slice_cube_window(
        mycube, attribute=attrs, sampling="trilinear", zsurf=sabove, other=sbelow
    )
    for attr in myattrs.keys():
        if DEBUG:
            myattrs[attr].describe()

        myattrs[attr].to_file("myfile_asymmetric_" + attr + ".dat", fformat="ijxyz")


def attribute_around_constant_cube_slices():
    """Get attribute around a constant cube slices"""

    cubefile = join(EXPATH1, "ib_test_cube2.segy")

    level1 = 1010
    level2 = 1100

    mycube = xtgeo.cube_from_file(cubefile)

    # instead of using zrange, we make some tmp surfaces that
    # reflects the assymmetric; here sample slices from cube
    sabove = xtgeo.surface_from_cube(mycube, level1)
    sbelow = xtgeo.surface_from_cube(mycube, level2)

    if DEBUG:
        sabove.describe()
        sbelow.describe()

    attrs = "all"

    myattrs = sabove.slice_cube_window(
        mycube, attribute=attrs, sampling="trilinear", zsurf=sabove, other=sbelow
    )
    for attr in myattrs.keys():
        if DEBUG:
            myattrs[attr].describe()

        myattrs[attr].to_file("myfile_constlevels_" + attr + ".dat", fformat="ijxyz")


if __name__ == "__main__":

    slice_a_cube_with_surface()
    attribute_around_surface_symmetric()
    attribute_around_surface_asymmetric()
    attribute_around_constant_cube_slices()

    print("Running example OK: {}".format(basename(__file__)))
