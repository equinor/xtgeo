import io
import concurrent.futures
import logging

import xtgeo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TESTFILE = "../../xtgeo-testdata/surfaces/reek/1/basereek_rota.gri"
NTHREAD = 20


def _get_files_as_regularsurfaces_thread(option=1):
    surfs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NTHREAD) as executor:

        if option == 1:
            futures = {executor.submit(_get_regsurff, i): i for i in range(NTHREAD)}
        else:
            futures = {executor.submit(_get_regsurfi, i): i for i in range(NTHREAD)}

        for future in concurrent.futures.as_completed(futures):
            try:
                surf = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error: %s", exc)
            else:
                surfs.append(surf)

        regular_surfaces = xtgeo.Surfaces(surfs)
        return regular_surfaces


def _get_files_as_regularsurfaces_multiprocess(option=1):
    surfs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NTHREAD) as executor:

        if option == 1:
            futures = {executor.submit(_get_regsurff, i): i for i in range(NTHREAD)}
        else:
            futures = {executor.submit(_get_regsurfi, i): i for i in range(NTHREAD)}

        for future in concurrent.futures.as_completed(futures):
            try:
                surf = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error: %s", exc)
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

    SURFS1 = _get_files_as_regularsurfaces_thread(option=1)
    for srf in SURFS1.surfaces:
        logger.info("1 %s", srf.values.mean())

    SURFS2 = _get_files_as_regularsurfaces_thread(option=2)
    for srf in SURFS2.surfaces:
        logger.info("2 %s", srf.values.mean())

    SURFS3 = _get_files_as_regularsurfaces_multiprocess(option=1)
    for srf in SURFS3.surfaces:
        logger.info("3 %s", srf.values.mean())

    SURFS4 = _get_files_as_regularsurfaces_multiprocess(option=2)
    for srf in SURFS4.surfaces:
        logger.info("4 %s", srf.values.mean())
