# -*- coding: utf-8 -*-
"""
Compute statistics within one realisation, using ROFF or RMS internal.
By JRIV
"""
from __future__ import division, absolute_import
from __future__ import print_function

from os.path import join as ojn
import xtgeo

EXPATH1 = "../../xtgeo-testdata/3dgrids/reek2"

ROOT = "geogrid"
EXT = ".roff"

GRIDFILE = ojn(EXPATH1, ROOT + EXT)

PROPS = ["perm", "poro"]
FACIES = "facies"
FACIESFILE = ojn(EXPATH1, ROOT + "--" + FACIES + EXT)


def show_stats():
    """Get statistics for one realisation, poro/perm filtered on facies.

    But note that values here are unweighted as total volume is not present.
    """

    # read grid
    grd = xtgeo.grid_from_file(GRIDFILE)

    # read facies (to be used as filter)
    facies = xtgeo.gridproperty_from_file(FACIESFILE, name=FACIES, grid=grd)
    print("Facies codes are: {}".format(facies.codes))

    for propname in PROPS:
        pfile = ojn(EXPATH1, ROOT + "--" + propname + EXT)
        pname = "geogrid--" + propname
        prop = xtgeo.gridproperty_from_file(pfile, name=pname, grid=grd)
        print("Working with {}".format(prop.name))

        # now find statistics for each facies, and all facies
        for key, fname in facies.codes.items():
            avg = prop.values[facies.values == key].mean()
            std = prop.values[facies.values == key].std()
            print(
                "For property {} in facies {}, avg is {:10.3f} and "
                "stddev is {:9.3f}".format(propname, fname, avg, std)
            )

        avg = prop.values.mean()
        std = prop.values.std()
        print(
            "For property {} in ALL facies, avg is {:10.3f} and "
            "stddev is {:9.3f}".format(propname, avg, std)
        )


def show_stats_inside_rms():
    """Get statistics for one realisation, poro/perm filtered on facies.

    This is an 'inside RMS' version; should work given runrmsx <project>
    but not tested. Focus on syntax for getting properties, otherwise code
    is quite similar.
    """

    prj = project  # pylint: disable=undefined-variable
    # names of icons...
    gridmodel = "Reek"
    faciesname = "Facies"
    propnames = ["Poro", "Perm"]

    # read facies (to be used as filter)
    facies = xtgeo.gridproperty_from_roxar(prj, gridmodel, faciesname)
    print("Facies codes are: {}".format(facies.codes))

    for propname in propnames:
        prop = xtgeo.gridproperty_from_roxar(prj, gridmodel, propname)
        print("Working with {}".format(prop.name))

        # now find statistics for each facies, and all facies
        for key, fname in facies.codes.items():
            avg = prop.values[facies.values == key].mean()
            std = prop.values[facies.values == key].std()
            print(
                "For property {} in facies {}, avg is {:10.3f} and "
                "stddev is {:9.3f}".format(propname, fname, avg, std)
            )

        avg = prop.values.mean()
        std = prop.values.std()
        print(
            "For property {} in ALL facies, avg is {:10.3f} and "
            "stddev is {:9.3f}".format(propname, avg, std)
        )


if __name__ == "__main__":

    show_stats()
    # show_stats_inside_rms()
