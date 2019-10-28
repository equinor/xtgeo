# coding: utf-8
from __future__ import division, absolute_import
import subprocess
import pytest
import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTFILE = "../xtgeo-testdata/surfaces/reek/1/basereek_rota_v2.gri"


@pytest.mark.skipif("sys.version_info < (3, 6)")
def test_surface_forks():
    """Testing when surfaces are read by multiple forks"""

    process = subprocess.Popen(
        ["python", "multiprocess_surfs.py"],
        cwd="examples",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    ret_code = process.wait()
    if ret_code:
        raise Exception(stderr)
    return stdout
