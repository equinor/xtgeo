# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import io
import concurrent.futures

import pytest
import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTFILE = "../xtgeo-testdata/surfaces/reek/1/basereek_rota_v2.gri"


def test_surface_forks():
    """Testing when surfaces are read by multiple forks"""

    def _get_files_as_regularsurfaces_thread(nthread, option=1):
        surfs = []

        logger.info("Number of threads are %s", nthread)
        with concurrent.futures.ThreadPoolExecutor(max_workers=nthread) as executor:

            if option == 1:
                futures = {executor.submit(_get_regsurff, i): i for i in range(nthread)}
            else:
                futures = {executor.submit(_get_regsurfi, i): i for i in range(nthread)}

            for future in concurrent.futures.as_completed(futures):
                try:
                    surf = future.result()
                except Exception as exc:
                    logger.error(f'Error: {exc}')
                else:
                    surfs.append(surf)

            regular_surfaces = xtgeo.Surfaces(surfs)
            return regular_surfaces

    def _get_regsurff(i):
        logger.info("Start %s", i)

        sfile = TESTFILE

        logger.info("File is %s", sfile)
        rf = xtgeo.RegularSurface(sfile)
        logger.info("End %s", i)
        return rf

    def _get_regsurfi(i):
        logger.info("Start %s", i)

        sfile = TESTFILE
        with open(sfile, "rb") as fin:
            stream = io.BytesIO(fin.read())

        logger.info("File is %s", sfile)
        rf = xtgeo.RegularSurface(stream, fformat="irap_binary")
        logger.info("End %s", i)

        return rf

    #
    # main:
    #

    nthread = 2

    surfs1 = _get_files_as_regularsurfaces_thread(nthread, option=1)
    for surf in surfs1.surfaces:
        assert surf.values.mean() == pytest.approx(1736.1, abs=0.1)

    surfs2 = _get_files_as_regularsurfaces_thread(nthread, option=2)
    for surf in surfs2.surfaces:
        assert surf.values.mean() == pytest.approx(1736.1, abs=0.1)
