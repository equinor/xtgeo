# -*- coding: utf-8 -*-
import os


import pytest
import xtgeo
from xtgeo import pathlib
import test_common.test_xtg as tsetup

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True


TESTFILE = "../xtgeo-testdata/3dgrids/reek/REEK.EGRID"
TESTFOLDER = "../xtgeo-testdata/3dgrids/reek"
TESTNOEXISTFILE = "../xtgeo-testdata/3dgrids/reek/NOSUCH.EGRID"
TESTNOEXISTFOLDER = "../xtgeo-testdata/3dgrids/noreek/NOSUCH.EGRID"


@tsetup.skipifmac
@tsetup.skipifwindows
def test_xtgeocfile():
    """Test basic system file io etc functions"""

    gfile = xtgeo._XTGeoFile(TESTFILE)
    xfile = xtgeo._XTGeoFile(TESTNOEXISTFILE)
    yfile = xtgeo._XTGeoFile(TESTNOEXISTFOLDER)
    gfolder = xtgeo._XTGeoFile(TESTFOLDER)

    assert isinstance(gfile, xtgeo._XTGeoFile)

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

    assert "Swig" in str(gfile.get_cfhandle())
    assert gfile.cfclose() is True

    # extensions:
    stem, suff = gfile.splitext(lower=False)
    assert stem == "REEK"
    assert suff == "EGRID"


@tsetup.skipifmac
@tsetup.skipifwindows
def test_xtgeocfile_fhandle():
    """Test in particular C handle SWIG system"""

    gfile = xtgeo._XTGeoFile(TESTFILE)

    chandle1 = gfile.get_cfhandle()
    chandle2 = gfile.get_cfhandle()
    assert gfile._cfhandlecount == 2
    assert chandle1 == chandle2
    assert gfile.cfclose() is False
    assert gfile.cfclose() is True

    # try to close a cfhandle that does not exist
    with pytest.raises(RuntimeError):
        gfile.cfclose()


# @tsetup.skipifwindows
# @tsetup.skipifpython2
# def test_xtgeocfile_bytesio():

#     with open(TESTFILE, "rb") as fin:
#         stream = io.BytesIO(fin.read())

#     gfile = xtgeo._XTGeoFile(stream)

#     assert isinstance(gfile, xtgeo._XTGeoFile)

#     assert "Swig" in str(gfile.fhandle)

#     assert gfile.close() is True


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
