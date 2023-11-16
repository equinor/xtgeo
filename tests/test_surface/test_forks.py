# coding: utf-8

import os
import subprocess

from xtgeo.common.xtgeo_dialog import testdatafolder


def test_surface_forks() -> None:
    """Testing when surfaces are read by multiple forks"""

    env = dict(os.environ) | {"XTG_TESTPATH": testdatafolder}
    process = subprocess.Popen(
        ["python", "multiprocess_surfs.py"],
        cwd="examples",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    _, stderr = process.communicate()
    ret_code = process.wait()
    if ret_code:
        raise Exception(stderr)
