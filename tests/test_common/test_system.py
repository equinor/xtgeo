# -*- coding: utf-8 -*-
import hashlib
import os
import pathlib

import pytest

import tests.test_common.test_xtg as tsetup
import xtgeo
import xtgeo.common.sys as xsys

xtg = xtgeo.XTGeoDialog()

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True

TPATH = xtg.testpathobj

TESTFILE = TPATH / "3dgrids/reek/REEK.EGRID"
TESTFOLDER = TPATH / "3dgrids/reek"
TESTNOEXISTFILE = TPATH / "3dgrids/reek/NOSUCH.EGRID"
TESTNOEXISTFOLDER = TPATH / "3dgrids/noreek/NOSUCH.EGRID"
TESTSURF = TPATH / "surfaces/reek/1/topupperreek.gri"


def test_generic_hash():
    """Testing generic hashlib function."""
    ahash = xsys.generic_hash("ABCDEF")
    assert ahash == "8827a41122a5028b9808c7bf84b9fcf6"

    ahash = xsys.generic_hash("ABCDEF", hashmethod="sha256")
    assert ahash == "e9c0f8b575cbfcb42ab3b78ecc87efa3b011d9a5d10b09fa4e96f240bf6a82f5"

    ahash = xsys.generic_hash("ABCDEF", hashmethod="blake2b")
    assert ahash[0:12] == "0bb3eb1511cb"

    with pytest.raises(KeyError):
        ahash = xsys.generic_hash("ABCDEF", hashmethod="invalid")

    # pass a hashlib function
    ahash = xsys.generic_hash("ABCDEF", hashmethod=hashlib.sha224)
    assert ahash == "fd6639af1cc457b72148d78e90df45df4d344ca3b66fa44598148ce4"


def test_resolve_alias():
    """Testing resolving file alias function."""
    surf = xtgeo.RegularSurface(TESTSURF)
    md5hash = surf.generate_hash("md5")

    mname = xtgeo._XTGeoFile("whatever/$md5sum.gri", obj=surf)
    assert str(mname.file) == f"whatever/{md5hash}.gri"

    mname = xtgeo._XTGeoFile(pathlib.Path("whatever/$md5sum.gri"), obj=surf)
    assert str(mname.file) == f"whatever/{md5hash}.gri"

    mname = xtgeo._XTGeoFile("whatever/$random.gri", obj=surf)
    assert len(str(mname.file)) == 45

    # use $fmu.v1 schema
    surf.metadata.opt.shortname = "topValysar"
    surf.metadata.opt.description = "Depth surface"

    mname = xtgeo._XTGeoFile(pathlib.Path("whatever/$fmu-v1.gri"), obj=surf)
    assert str(mname.file) == "whatever/topvalysar--depth_surface.gri"


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

    with pytest.raises(OSError):
        xfile.check_file(raiseerror=OSError)

    assert gfile.check_folder() is True
    assert xfile.check_folder() is True
    assert yfile.check_folder() is False
    with pytest.raises(OSError):
        yfile.check_folder(raiseerror=OSError)

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
