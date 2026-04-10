"""Shared fixtures for testing the ResInsight interface.

Shall only depend on the ResInsight interface layer
(xtgeo.interfaces.resinsight), not on other xtgeo modules."""

import logging
import pathlib

import pytest

from xtgeo.interfaces.resinsight._rips_package import RipsInstanceType
from xtgeo.interfaces.resinsight.rips_utils import RipsApiUtils

DROGON_GRID = pathlib.Path("3dgrids/drogon/2/geogrid.roff")
EMERALD_GRID = pathlib.Path("3dgrids/eme/1/emerald.roff")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@pytest.fixture(scope="module")
def resinsight_instance(testdata_path) -> RipsInstanceType:
    """Create a ResInsight instance for testing.

    Use console_mode to avoid opening the full GUI, which is not needed for API tests
    and can cause issues in headless environments.
    """
    pytest.importorskip(
        "rips", reason="ResInsight API tests require 'rips' package to be installed"
    )
    logger.info("Creating ResInsight instance for testing")
    try:
        instance = RipsApiUtils.launch_instance(executable="", console_mode=True)
    except (RuntimeError, OSError) as e:
        pytest.skip(
            f"ResInsight executable not available (set RESINSIGHT_EXECUTABLE env var "
            f"or add ResInsight to PATH): {e}"
        )
    path = pathlib.Path(testdata_path)

    drogon = instance.project.load_case(path=str(path / DROGON_GRID))
    emerald = instance.project.load_case(path=str(path / EMERALD_GRID))

    # Give the cases with same name to test 'find_last' functionality
    # in GridReader/GridWriter
    drogon.name = "EXAMPLE"
    drogon.update()
    emerald.name = "EXAMPLE"
    emerald.update()

    logger.info("ResInsight instance created and test cases loaded")
    yield instance

    # Teardown: close the instance after tests are done
    logger.info("Closing ResInsight instance after testing")
    instance.exit()
