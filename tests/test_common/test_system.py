# -*- coding: utf-8 -*-
import hashlib
import os
import pathlib
import io

import pytest

import tests.test_common.test_xtg as tsetup
import xtgeo
import xtgeo.common.sys as xsys

xtg = xtgeo.XTGeoDialog()

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True

TPATH = xtg.testpathobj
TMPD = xtg.tmpdirobj

TEST_ECL_ROOT = TPATH / "3dgrids/reek/REEK"
TESTFILE = TPATH / "3dgrids/reek/REEK.EGRID"
TESTFOLDER = TPATH / "3dgrids/reek"
TESTNOEXISTFILE = TPATH / "3dgrids/reek/NOSUCH.EGRID"
TESTNOEXISTFOLDER = TPATH / "3dgrids/noreek/NOSUCH.EGRID"
TESTSURF = TPATH / "surfaces/reek/1/topupperreek.gri"
TESTSURF2 = TPATH / "surfaces/reek/1/reek_stooip_map.gri"
TESTSURF3 = TPATH / "surfaces/etc/seabed_p.pmd"
TESTROFFGRIDB = TPATH / "3dgrids/reek/reek_geo_grid.roff"
TESTROFFGRIDA = TPATH / "3dgrids/reek/reek_geogrid.roffasc"
TESTWELL1 = TPATH / "wells/battle/1/WELL12.rmswell"
MDFILE = TPATH / "README.md"  # to test a file not relevant for fformat


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
    """Test basic system file io etc functions."""
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
    """Test in particular C handle SWIG system."""

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


def test_detect_fformat():
    """Test to guess/detect file formats based on various criteria."""
    # irap binary as file
    gfile = xtgeo._XTGeoFile(TESTSURF2)
    assert gfile.detect_fformat() == "irap_binary"

    # irap binary as memory stream
    stream = io.BytesIO()
    surf = xtgeo.RegularSurface(TESTSURF2)
    surf.to_file(stream)
    sfile = xtgeo._XTGeoFile(stream)
    assert sfile.memstream is True
    assert sfile.detect_fformat() == "irap_binary"

    # petromod binary as file
    gfile = xtgeo._XTGeoFile(TESTSURF3)
    assert gfile.detect_fformat() == "petromod"

    # HDF xtgeo surface file
    newfile = TMPD / "hdf_surf.hdf"
    surf.to_hdf(newfile)
    gfile = xtgeo._XTGeoFile(newfile)
    assert gfile.detect_fformat() == "hdf"
    assert gfile.detect_fformat(details=True) == "hdf RegularSurface xtgeo"

    # HDF xtgeo surface as stream
    stream = io.BytesIO()
    surf = xtgeo.RegularSurface(TESTSURF2)
    surf.to_hdf(stream)
    sfile = xtgeo._XTGeoFile(stream)
    assert sfile.memstream is True
    assert sfile.detect_fformat() == "hdf"

    # Eclipse egrid as file
    gfile = xtgeo._XTGeoFile(TEST_ECL_ROOT.with_suffix(".EGRID"))
    assert gfile.detect_fformat() == "egrid"
    # Eclipse restart (unified) as file
    gfile = xtgeo._XTGeoFile(TEST_ECL_ROOT.with_suffix(".UNRST"))
    assert gfile.detect_fformat() == "unrst"
    # Eclipse init as file
    gfile = xtgeo._XTGeoFile(TEST_ECL_ROOT.with_suffix(".INIT"))
    assert gfile.detect_fformat() == "init"
    # Eclipse init as file, extension only
    gfile = xtgeo._XTGeoFile(TEST_ECL_ROOT.with_suffix(".INIT"))
    assert gfile.detect_fformat(suffixonly=True) == "init"

    # ROFF bin as file
    gfile = xtgeo._XTGeoFile(TESTROFFGRIDB)
    assert gfile.detect_fformat() == "roff_binary"
    # ROFF asc as file
    gfile = xtgeo._XTGeoFile(TESTROFFGRIDA)
    assert gfile.detect_fformat() == "roff_ascii"

    # RMS ascii well file
    wfile = xtgeo._XTGeoFile(TESTWELL1)
    assert wfile.detect_fformat() == "rmswell"

    # RMS ascii well file by suffix only
    wfile = xtgeo._XTGeoFile(TESTWELL1)
    assert wfile.detect_fformat(suffixonly=True) == "rmswell"

    # Some invalid case
    wfile = xtgeo._XTGeoFile(MDFILE)
    assert wfile.detect_fformat(suffixonly=True) == "unknown"


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
