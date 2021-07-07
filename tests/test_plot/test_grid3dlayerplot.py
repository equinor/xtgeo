# -*- coding: utf-8 -*-


import os

import matplotlib.pyplot as plt
import pytest

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

if "XTG_SHOW" in os.environ:
    XTGSHOW = True
else:
    XTGSHOW = False

logger.info("Use env variable XTG_SHOW to show interactive plots to screen")

TPATH = xtg.testpathobj

USEFILE1 = TPATH / "3dgrids/reek/reek_sim_grid.roff"
USEFILE2 = TPATH / "3dgrids/reek/reek_sim_poro.roff"
USEFILE3 = TPATH / "etc/colortables/rainbow_reverse.rmscolor"


@pytest.mark.skipifroxar
def test_very_basic(tmpdir):
    """Just test that matplotlib works."""
    assert "matplotlib" in str(plt)

    plt.title("Hello world")
    plt.savefig(os.path.join(tmpdir, "helloworld1.png"))
    plt.savefig(os.path.join(tmpdir, "helloworld1.svg"))
    if XTGSHOW:
        plt.show()
    logger.info("Very basic plotting")
    plt.close()
