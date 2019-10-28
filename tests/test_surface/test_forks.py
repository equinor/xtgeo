# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
import concurrent.futures

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir


def test_surface_forks():
    """Testing when surfaces are read by multiple forks"""

    def _get_files_as_regularsurfaces_thread(nthread):
        surfs = []

        logger.info("Number of threads are %s", nthread)
        with concurrent.futures.ThreadPoolExecutor(max_workers=nthread) as executor:

            futures = {executor.submit(_get_regsurf, i): i for i in range(nthread)}

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

        sfile = os.path.join(TMPD, str(i) + ".gri")
        # with open(file, "rb") as fin:
        #     stream = io.BytesIO(fin.read())

        logger.info("File is %s", sfile)
        rf = xtgeo.RegularSurface(sfile)
        logger.info("End %s", i)
        return rf

    #
    # main:
    #
    nthread = 100
    for n in range(nthread):
        x = xtgeo.RegularSurface()
        x.to_file(os.path.join(TMPD, str(n) + ".gri"))

    # _get_files_as_regularsurfaces_thread(nthread)
