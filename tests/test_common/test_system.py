# -*- coding: utf-8 -*-
import os
import io

import xtgeo
import test_common.test_xtg as tsetup

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True


TESTFILE = "../xtgeo-testdata/3dgrids/reek/REEK.EGRID"
# =============================================================================
# Do tests of simple system functions
# =============================================================================


def test_xtgeocfile():

    gfile = xtgeo._XTGeoCFile(TESTFILE)
    assert isinstance(gfile, xtgeo._XTGeoCFile)

    assert "Swig" in str(gfile.fhandle)

    assert gfile.close() is True


#@tsetup.skipifwindows
def test_xtgeocfile_bytesio():

    with open(TESTFILE, "rb") as fin:
        stream = io.BytesIO(fin.read())

    gfile = xtgeo._XTGeoCFile(stream)

    assert isinstance(gfile, xtgeo._XTGeoCFile)

    assert "Swig" in str(gfile.fhandle)

    assert gfile.close() is True


# @tsetup.equinor
# def test_check_folder():
#     """testing that folder checks works in different scenaria"""

#     status = xsys.check_folder("setup.py")
#     assert status is True

#     status = xsys.check_folder("xxxxx/whatever")
#     assert status is False

#     status = xsys.check_folder("src/xtgeo")
#     assert status is True

#     if "WINDOWS" in platform.system().upper():
#         return
#     if not TRAVIS:
#         print("Non travis test")
#         # skipped for travis, as travis runs with root rights
#         folder = "TMP/nonwritable"
#         myfile = os.path.join(folder, "somefile")
#         if not os.path.exists(folder):
#             os.mkdir(folder, 0o440)

#         status = xsys.check_folder(myfile)
#         assert status is False

#         status = xsys.check_folder(folder)
#         assert status is False

#         with pytest.raises(ValueError):
#             xsys.check_folder(folder, raiseerror=ValueError)
