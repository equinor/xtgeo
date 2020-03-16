# -*- coding: utf-8 -*-
import os
import io

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

import pytest
import xtgeo
import test_common.test_xtg as tsetup

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True


TESTFILE = "../xtgeo-testdata/3dgrids/reek/REEK.EGRID"
TESTFOLDER = "../xtgeo-testdata/3dgrids/reek"
TESTNOEXISTFILE = "../xtgeo-testdata/3dgrids/reek/NOSUCH.EGRID"
TESTNOEXISTFOLDER = "../xtgeo-testdata/3dgrids/noreek/NOSUCH.EGRID"


def test_xtgeocfile():
    """Test basic system file io etc functions"""

    gfile = xtgeo._XTGeoCFile(TESTFILE)
    xfile = xtgeo._XTGeoCFile(TESTNOEXISTFILE)
    yfile = xtgeo._XTGeoCFile(TESTNOEXISTFOLDER)
    gfolder = xtgeo._XTGeoCFile(TESTFOLDER)

    assert isinstance(gfile, xtgeo._XTGeoCFile)

    assert isinstance(gfile._file, pathlib.Path)

    assert gfile._memstream is False
    assert gfile._mode == "rb"
    assert gfile._delete_after is False
    assert gfile.name == os.path.abspath(TESTFILE)
    assert xfile.name == os.path.abspath(TESTNOEXISTFILE)

    # exists, check_*
    assert gfile.exists() is True
    assert gfolder.exists() is True
    assert xfile.exists() is False

    assert gfile.check_file() is True
    assert xfile.check_file() is False
    assert yfile.check_file() is False

    with pytest.raises(IOError):
        xfile.check_file(raiseerror=IOError)

    assert gfile.check_folder() is True
    assert xfile.check_folder() is True
    assert yfile.check_folder() is False
    with pytest.raises(IOError):
        yfile.check_folder(raiseerror=IOError)

    assert "Swig" in str(gfile.fhandle)
    assert gfile.close() is True

    # extensions:
    stem, suff = gfile.splitext(lower=False)
    assert stem == "REEK"
    assert suff == "EGRID"


@tsetup.skipifwindows
@tsetup.skipifpython2
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
