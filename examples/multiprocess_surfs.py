import io
import concurrent.futures
import logging

import xtgeo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TESTFILE = "../../xtgeo-testdata/surfaces/reek/1/basereek_rota.gri"


def _get_files_as_regularsurfaces_thread(nthread, option=1):
    surfs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=nthread) as executor:

        if option == 1:
            futures = {executor.submit(_get_regsurff, i): i for i in range(nthread)}
        else:
            futures = {executor.submit(_get_regsurfi, i): i for i in range(nthread)}

        for future in concurrent.futures.as_completed(futures):
            try:
                surf = future.result()
            except Exception as exc:
                logger.error("Error: ", exc)
            else:
                surfs.append(surf)

        regular_surfaces = xtgeo.Surfaces(surfs)
        return regular_surfaces


def _get_files_as_regularsurfaces_multiprocess(nthread, option=1):
    surfs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=nthread) as executor:

        if option == 1:
            futures = {executor.submit(_get_regsurff, i): i for i in range(nthread)}
        else:
            futures = {executor.submit(_get_regsurfi, i): i for i in range(nthread)}

        for future in concurrent.futures.as_completed(futures):
            try:
                surf = future.result()
            except Exception as exc:
                logger.error("Error: ", exc)
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


if __name__ == "__main__":

    nthread = 20

    surfs1 = _get_files_as_regularsurfaces_thread(nthread, option=1)
    for surf in surfs1.surfaces:
        logger.info("1 %s", surf.values.mean())

    surfs2 = _get_files_as_regularsurfaces_thread(nthread, option=2)
    for surf in surfs2.surfaces:
        logger.info("2 %s", surf.values.mean())

    surfs3 = _get_files_as_regularsurfaces_multiprocess(nthread, option=1)
    for surf in surfs3.surfaces:
        logger.info("3 %s", surf.values.mean())

    surfs4 = _get_files_as_regularsurfaces_multiprocess(nthread, option=2)
    for surf in surfs4.surfaces:
        logger.info("4 %s", surf.values.mean())
