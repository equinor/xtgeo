# coding: utf-8

import subprocess

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
from xtgeo.common import logger
from xtgeo.common.xtgeo_dialog import testdatafolder

TPATH = testdatafolder

TESTFILE = TPATH / "surfaces/reek/1/basereek_rota_v2.gri"


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
