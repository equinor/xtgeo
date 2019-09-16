# -*- coding: utf-8 -*-
import os
import pytest
import xtgeo.common.xtgeo_system as xsys
import platform

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True

# =============================================================================
# Do tests of simple system functions
# =============================================================================


def test_check_folder():
    """testing that folder checks works in different scenaria"""

    status = xsys.check_folder("setup.py")
    assert status is True

    status = xsys.check_folder("xxxxx/whatever")
    assert status is False

    status = xsys.check_folder("src/xtgeo")
    assert status is True

    if "WINDOWS" in platform.system().upper():
        return
    if not TRAVIS:
        print("Non travis test")
        # skipped for travis, as travis runs with root rights
        folder = "TMP/nonwritable"
        myfile = os.path.join(folder, "somefile")
        if not os.path.exists(folder):
            try:
                os.mkdir(folder, mode=0o440)
            except TypeError:
                os.mkdir(folder, 0440)

        status = xsys.check_folder(myfile)
        assert status is False

        status = xsys.check_folder(folder)
        assert status is False

        with pytest.raises(ValueError):
            xsys.check_folder(folder, raiseerror=ValueError)
