import os
import os.path
import sys
import logging

import xtgeo.common.calc as xcalc
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()
logging.basicConfig(format=xtg.loggingformat, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)

logger = logging.getLogger(__name__)

# =============================================================================
# Do tests
# =============================================================================


def test_ijk_to_ib():
    """Convert I J K to IB index."""

    ib = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5)
    logger.info(ib)
    assert ib == 16


def test_ib_to_ijk():
    """Convert IB index to IJK tuple."""

    ijk = xcalc.ib_to_ijk(16, 3, 4, 5)
    logger.info(ijk)
    assert ijk[0] == 2
