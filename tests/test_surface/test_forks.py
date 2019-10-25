# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import math
from os.path import join

import concurrent.futures
import io

import xtgeo
import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath

TESTSET1A = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"

def test_surface_forks():
    """Testing when surfaces are read by multiple forks"""

    def _get_files_as_regularsurfaces_thread(nthread):
        surfs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=nthread) as executor:

            futures = {executor.submit(_get_regsurf, i): i for i in range(0, nthread)}

            for future in concurrent.futures.as_completed(futures):
                try:
                    surf = future.result()
                except Exception as exc:
                    logger.error(f'Error: {exc}')
                else:
                    surfs.append(surf)

            regular_surfaces = xtgeo.Surfaces(surfs)
            return regular_surfaces


    def _get_regsurf(i):
        logger.info("Start %s", i)

        sfile = TESTSET1A
        # with open(file, "rb") as fin:
        #     stream = io.BytesIO(fin.read())

        rf = xtgeo.RegularSurface(sfile)
        logger.info("End %s", i)
        return rf

    _get_files_as_regularsurfaces_thread(1)
