# coding: utf-8

import subprocess

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)


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
