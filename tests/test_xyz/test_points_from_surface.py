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
    assert poi.get_dataframe()[poi.zname][0] == 1234.0

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.get_dataframe(), surf.get_dataframe())

    poi2 = xtgeo.points_from_surface(surf, zname="VALUES")

    pd.testing.assert_frame_equal(poi2.get_dataframe(), surf.get_dataframe())


def test_init_with_surface_classmethod(testdata_path):
    """Initialise points object with surface instance."""
    surf = xtgeo.surface_from_file(testdata_path / SURFACE)
    poi = xtgeo.points_from_surface(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.get_dataframe(), surf.get_dataframe())
