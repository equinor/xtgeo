import concurrent.futures
import io
import logging

import xtgeo
from xtgeo.common.xtgeo_dialog import testdatafolder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NTHREAD = 20


def _get_files_as_regularsurfaces_thread(option: int = 1) -> xtgeo.Surfaces:
    surfs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NTHREAD) as executor:
        if option == 1:
            futures = {executor.submit(_get_regsurff, i): i for i in range(NTHREAD)}
        else:
            futures = {executor.submit(_get_regsurfi, i): i for i in range(NTHREAD)}

        for future in concurrent.futures.as_completed(futures):
            try:
                surfs.append(future.result())
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error: %s", exc)

    return xtgeo.Surfaces(surfs)


def _get_files_as_regularsurfaces_multiprocess(option: int = 1) -> xtgeo.Surfaces:
    surfs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NTHREAD) as executor:
        if option == 1:
            futures = {executor.submit(_get_regsurff, i): i for i in range(NTHREAD)}
        else:
            futures = {executor.submit(_get_regsurfi, i): i for i in range(NTHREAD)}

        for future in concurrent.futures.as_completed(futures):
            try:
                surfs.append(future.result())
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error: %s", exc)

    return xtgeo.Surfaces(surfs)


def _get_regsurff(i: int) -> xtgeo.Surfaces:
    logger.info("Start %s", i)
    logger.info("File is %s", testdatafolder)
    rf = xtgeo.surface_from_file(testdatafolder)
    logger.info("End %s", i)
    return rf


def _get_regsurfi(i: int) -> xtgeo.Surfaces:
    logger.info("Start %s", i)
    with open(testdatafolder, "rb") as fin:
        stream = io.BytesIO(fin.read())

    logger.info("File is %s", testdatafolder)
    rf = xtgeo.surface_from_file(stream, fformat="irap_binary")
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
