import io
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pytest

import xtgeo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_regsurf_from_file(surf_path: Path, n: int) -> xtgeo.RegularSurface:
    logger.info("Start %s", n)
    rf = xtgeo.surface_from_file(surf_path, fformat="irap_binary")
    logger.info("End %s", n)
    return rf


def _get_regsurf_from_stream(surf_path: Path, n: int) -> xtgeo.RegularSurface:
    logger.info("Start %s", n)
    with open(surf_path, "rb") as fin:
        stream = io.BytesIO(fin.read())
    rf = xtgeo.surface_from_file(stream, fformat="irap_binary")
    logger.info("End %s", n)
    return rf


def run_concurrent(
    executor_class: type[ProcessPoolExecutor] | type[ThreadPoolExecutor],
    loader_func: Callable[[Path, int], xtgeo.RegularSurface],
    surf_path: Path,
    n_concurrent: int,
) -> list[xtgeo.RegularSurface]:
    """Run concurrent loading with executor and loader."""
    surfs = []
    with executor_class(max_workers=n_concurrent) as executor:
        futures = {
            executor.submit(loader_func, surf_path, n): n for n in range(n_concurrent)
        }
        for future in as_completed(futures):
            surf = future.result()
            surfs.append(surf)
    return surfs


@pytest.mark.parametrize(
    "executor_class, loader_func",
    [
        (ThreadPoolExecutor, _get_regsurf_from_file),
        (ThreadPoolExecutor, _get_regsurf_from_stream),
        (ProcessPoolExecutor, _get_regsurf_from_file),
        (ProcessPoolExecutor, _get_regsurf_from_stream),
    ],
)
def test_concurrent_surface_loading(
    executor_class: type[ProcessPoolExecutor] | type[ThreadPoolExecutor],
    loader_func: Callable[[Path, int], xtgeo.RegularSurface],
    testdata_path: str,
) -> None:
    """Test concurrent loading of surfaces using different executors and methods."""
    surf_path = Path(testdata_path) / "surfaces/reek/1/basereek_rota.gri"
    n_concurrent = 10
    surfs = run_concurrent(executor_class, loader_func, surf_path, n_concurrent)

    assert len(surfs) == n_concurrent
    surfaces = xtgeo.Surfaces(surfs)
    for srf in surfaces.surfaces:
        assert srf is not None
