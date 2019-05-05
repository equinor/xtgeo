# coding: utf-8
"""Just a few simple tests without using the xtgeo-testdata repo"""

from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo

xtg = xtgeo.common.XTGeoDialog()


def test_regular_surface():
    """Do tests on default surface"""

    srf = xtgeo.surface.RegularSurface()
    assert srf.ncol == 5
    assert srf.nrow == 3
