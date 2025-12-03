import io
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import xtgeo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TESTFILE = (
    Path(__file__).parent.parent.parent
    / "xtgeo-testdata/surfaces/reek/1/basereek_rota.gri"
)
NTHREAD = 20


def _get_regsurf_from_file(n: int) -> xtgeo.RegularSurface:
    logger.info("Start %s", n)
    rf = xtgeo.surface_from_file(TESTFILE, fformat="irap_binary")
    logger.info("End %s", n)
    return rf


def _get_regsurf_from_stream(n: int) -> xtgeo.RegularSurface:
    logger.info("Start %s", n)
    with open(TESTFILE, "rb") as fin:
        stream = io.BytesIO(fin.read())
    rf = xtgeo.surface_from_file(stream, fformat="irap_binary")
    logger.info("End %s", n)
    return rf


def run_test(
    executor_class: type[ProcessPoolExecutor] | type[ThreadPoolExecutor],
    loader_func: Callable[[int], xtgeo.RegularSurface],
    name: str,
) -> bool:
    """Generic test runner."""
    surfs = []
    with executor_class(max_workers=NTHREAD) as executor:
        futures = {executor.submit(loader_func, n): n for n in range(NTHREAD)}

        for future in as_completed(futures):
            surf = future.result()
            surfs.append(surf)

    surfaces = xtgeo.Surfaces(surfs)
    for srf in surfaces.surfaces:
        logger.info("%s: %s", name, srf.values.mean())

    return len(surfs) == NTHREAD


def main() -> int:
    """Main entry point."""

    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)

    success = True
    success &= run_test(ThreadPoolExecutor, _get_regsurf_from_file, "thread-file")
    success &= run_test(ThreadPoolExecutor, _get_regsurf_from_stream, "thread-stream")
    success &= run_test(ProcessPoolExecutor, _get_regsurf_from_file, "process-file")
    success &= run_test(ProcessPoolExecutor, _get_regsurf_from_stream, "process-stream")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
