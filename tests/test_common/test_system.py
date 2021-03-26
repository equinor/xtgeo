# -*- coding: utf-8 -*-
import hashlib
import io
import pathlib
import sys

import pytest

import xtgeo
import xtgeo.common.sys as xsys

xtg = xtgeo.XTGeoDialog()


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


surface_files_formats = {
    pathlib.Path("surfaces/reek/1/topupperreek.gri"): "irap_binary",
    pathlib.Path("surfaces/reek/1/reek_stooip_map.gri"): "irap_binary",
    pathlib.Path("surfaces/etc/seabed_p.pmd"): "petromod",
}


@pytest.mark.skipif(sys.platform == "win32", reason="Path delimiter issue")
@pytest.mark.parametrize("filename", surface_files_formats.keys())
def test_resolve_alias(testpath, filename):
    """Testing resolving file alias function."""
    surf = xtgeo.RegularSurface(testpath / filename)
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


@pytest.fixture(name="reek_grid_path")
def fixture_reek_grid_path(testpath):
    return pathlib.Path(testpath) / "3dgrids/reek"


@pytest.mark.parametrize("filename", ["REEK.EGRID", "."])
def test_file_does_exist(reek_grid_path, filename):
    xtgeo_file = xtgeo._XTGeoFile(reek_grid_path / filename)
    assert xtgeo_file.exists() is True


@pytest.mark.parametrize("filename", ["NOSUCH.EGRID", "NOSUCH/NOSUCH.EGRID"])
def test_file_does_not_exist(reek_grid_path, filename):
    xtgeo_file = xtgeo._XTGeoFile(reek_grid_path / filename)
    assert xtgeo_file.exists() is False


@pytest.mark.parametrize("filename", ["REEK.EGRID", "REEK.INIT"])
def test_check_file_is_ok(reek_grid_path, filename):
    xtgeo_file = xtgeo._XTGeoFile(reek_grid_path / filename)
    assert xtgeo_file.check_file() is True


@pytest.mark.parametrize("filename", ["NOSUCH.EGRID", "NOSUCH/NOSUCH.EGRID"])
def test_check_file(reek_grid_path, filename):
    xtgeo_file = xtgeo._XTGeoFile(reek_grid_path / filename)
    assert xtgeo_file.check_file() is False

    with pytest.raises(OSError):
        xtgeo_file.check_file(raiseerror=OSError)


@pytest.mark.parametrize(
    "filename, stem, extension",
    [
        (pathlib.Path("3dgrids/reek/REEK.EGRID"), "REEK", "EGRID"),
        (pathlib.Path("/tmp/text.txt"), "text", "txt"),
        (pathlib.Path("/tmp/null"), "null", ""),
    ],
)
def test_file_splitext(filename, stem, extension):
    xtgeo_file = xtgeo._XTGeoFile(filename)
    assert (stem, extension) == xtgeo_file.splitext(lower=False)


files_formats = {
    **surface_files_formats,
    **{
        pathlib.Path("3dgrids/reek/REEK.EGRID"): "egrid",
        pathlib.Path("3dgrids/reek/REEK.UNRST"): "unrst",
        pathlib.Path("3dgrids/reek/REEK.INIT"): "init",
        pathlib.Path("3dgrids/reek/reek_geo_grid.roff"): "roff_binary",
        pathlib.Path("3dgrids/reek/reek_geogrid.roffasc"): "roff_ascii",
        pathlib.Path("wells/battle/1/WELL12.rmswell"): "rmswell",
    },
}


@pytest.mark.parametrize("filename", files_formats.keys())
def xtgeo_file_properties(testpath, filename):
    gfile = xtgeo._XTGeoFile(testpath / filename)

    assert isinstance(gfile, xtgeo._XTGeoFile)
    assert isinstance(gfile._file, pathlib.Path)

    assert gfile._memstream is False
    assert gfile._mode == "rb"
    assert gfile._delete_after is False
    assert gfile.name == (testpath / filename).absolute()

    assert "Swig" in str(gfile.get_cfhandle())
    assert gfile.cfclose() is True


@pytest.mark.parametrize("filename", files_formats.keys())
def test_file_c_handle(testpath, filename):
    any_xtgeo_file = xtgeo._XTGeoFile(testpath / filename)

    handle_count = any_xtgeo_file._cfhandlecount

    c_handle_1 = any_xtgeo_file.get_cfhandle()
    assert handle_count + 1 == any_xtgeo_file._cfhandlecount

    c_handle_2 = any_xtgeo_file.get_cfhandle()
    assert handle_count + 2 == any_xtgeo_file._cfhandlecount

    assert c_handle_1 == c_handle_2

    assert any_xtgeo_file.cfclose() is False
    assert any_xtgeo_file.cfclose() is True

    # try to close a cfhandle that does not exist
    with pytest.raises(RuntimeError):
        any_xtgeo_file.cfclose()


@pytest.mark.parametrize("filename", surface_files_formats.keys())
def test_surface_file_roundtrip_stream(testpath, filename):
    stream = io.BytesIO()
    surf = xtgeo.RegularSurface(testpath / filename)
    surf.to_file(stream)
    stream_file = xtgeo._XTGeoFile(stream)

    assert stream_file.memstream is True
    assert stream_file.detect_fformat() == "irap_binary"


@pytest.mark.parametrize("filename, expected_format", files_formats.items())
def test_detect_fformat(testpath, filename, expected_format):
    xtgeo_file = xtgeo._XTGeoFile(testpath / filename)
    assert xtgeo_file.detect_fformat() == expected_format


@pytest.mark.parametrize("filename", surface_files_formats.keys())
def test_detect_fformat_hdf_stream(testpath, filename):
    stream = io.BytesIO()
    surf = xtgeo.RegularSurface(testpath / filename)
    surf.to_hdf(stream)
    sfile = xtgeo._XTGeoFile(stream)
    assert sfile.memstream is True
    assert sfile.detect_fformat() == "hdf"


@pytest.mark.parametrize("filename", surface_files_formats.keys())
def test_detect_fformat_hdf_to_file(tmp_path, testpath, filename):
    newfile = tmp_path / "hdf_surf.hdf"
    surf = xtgeo.RegularSurface(testpath / filename)
    surf.to_hdf(newfile)
    gfile = xtgeo._XTGeoFile(newfile)
    assert gfile.detect_fformat() == "hdf"
    assert gfile.detect_fformat(details=True) == "hdf RegularSurface xtgeo"


@pytest.mark.parametrize(
    "filename, expected_format",
    list(files_formats.items()) + [(pathlib.Path("README.md"), "unknown")],
)
def test_detect_fformat_suffix_only(testpath, filename, expected_format):
    xtgeo_file = xtgeo._XTGeoFile(testpath / filename)
    assert xtgeo_file.detect_fformat(suffixonly=True) == expected_format
