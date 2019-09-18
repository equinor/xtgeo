# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
from os.path import join
from collections import OrderedDict
import math

import pytest

import xtgeo
from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TESTPATH = xtg.testpath

DUALFILE = "../xtgeo-testdata/3dgrids/etc/COARSE_SCALE_2_E100"

# =============================================================================
# Do tests
# =============================================================================


def test_import_dualporo_grid():
    """Test grid with flag for dual porosity setup"""

    grd = xtgeo.grid_from_file(DUALFILE + ".EGRID")

    assert grd.dualporo is True

    # poro = xtgeo.gridproperty_from_file(DUALFILE + ".INIT", grid=grd, name="PORO")
