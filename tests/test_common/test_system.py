# -*- coding: utf-8 -*-
import os
import pytest
import xtgeo.common.xtgeo_system as xsys

TRAVIS = os.environ.get("TRAVIS", False)

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

    if not TRAVIS:
        # skipped for travis, as travis runs with root rights
        folder = "TMP/nonwritable"
        myfile = os.path.join(folder, "somefile")
        if not os.path.exists(folder):
            os.mkdir(folder, mode=0o440)
        status = xsys.check_folder(myfile)
        assert status is False

        status = xsys.check_folder(folder)
        assert status is False

        with pytest.raises(ValueError):
            xsys.check_folder(folder, raiseerror=ValueError)
