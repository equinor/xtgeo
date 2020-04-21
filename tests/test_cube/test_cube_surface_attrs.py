# -*- coding: utf-8 -*-
from os.path import join

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit("Cannot find test setup")

TMD = xtg.tmpdir
TPATH = xtg.testpath

SFILE1 = join(TPATH, "cubes/etc/ib_synth_iainb.segy")


@pytest.fixture()
def loadsfile1():
    """Fixture for loading a SFILE1"""
    logger.info("Load seismic file 1")
    return Cube(SFILE1)


def test_create():
    pass
