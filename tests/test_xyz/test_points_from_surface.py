import pathlib

import pandas as pd
import xtgeo

SURFACE = pathlib.Path("surfaces/reek/1/topreek_rota.gri")


def test_from_simple_surface():
    """Create points from a simple surface."""
    surf = xtgeo.RegularSurface(
        ncol=4, nrow=5, xori=0, yori=0, xinc=25, yinc=25, values=1234.0
    )
    poi = xtgeo.points_from_surface(surf)
    assert poi.dataframe[poi.zname][0] == 1234.0

    # old interface, to be deprecated from 2.16
    poi = xtgeo.Points()
    poi.from_surface(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())

    poi2 = xtgeo.Points()
    poi2.from_surface(surf, zname="VALUES")

    pd.testing.assert_frame_equal(poi2.dataframe, surf.dataframe())


def test_init_with_surface_deprecated(testpath):
    """Initialise points object with surface instance, to be deprecated."""
    surf = xtgeo.surface_from_file(testpath / SURFACE)
    poi = xtgeo.Points(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())


def test_init_with_surface_classmethod(testpath):
    """Initialise points object with surface instance."""
    surf = xtgeo.surface_from_file(testpath / SURFACE)
    poi = xtgeo.points_from_surface(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())
