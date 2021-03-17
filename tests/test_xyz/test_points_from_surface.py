import pandas as pd

from xtgeo import RegularSurface, Points


def test_from_simple_surface():
    """Create points from a simple surface."""

    surf = RegularSurface(ncol=4, nrow=5, xori=0, yori=0, xinc=25, yinc=25)
    poi = Points()
    poi.from_surface(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())

    poi = Points()
    poi.from_surface(surf, zname="VALUES")

    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())


def test_init_with_surface():
    """Initialise points object with surface instance."""

    surf = RegularSurface(ncol=4, nrow=5, xori=0, yori=0, xinc=25, yinc=25)
    poi = Points(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())
