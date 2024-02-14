import io
import pathlib
import sys
from collections import ChainMap

import pytest
import xtgeo
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.io._file import FileFormat, FileWrapper

xtg = xtgeo.XTGeoDialog()


SURFACE_FILE_FORMATS = {
    pathlib.Path("surfaces/reek/1/topupperreek.gri"): FileFormat.IRAP_BINARY,
    pathlib.Path("surfaces/reek/1/reek_stooip_map.gri"): FileFormat.IRAP_BINARY,
    pathlib.Path("surfaces/etc/seabed_p.pmd"): FileFormat.PETROMOD,
}

FILE_FORMATS = ChainMap(
    SURFACE_FILE_FORMATS,
    {
        pathlib.Path("3dgrids/reek/REEK.EGRID"): FileFormat.EGRID,
        pathlib.Path("3dgrids/reek/REEK.UNRST"): FileFormat.UNRST,
        pathlib.Path("3dgrids/reek/REEK.INIT"): FileFormat.INIT,
        pathlib.Path("3dgrids/reek/reek_geo_grid.roff"): FileFormat.ROFF_BINARY,
        pathlib.Path("3dgrids/reek/reek_geogrid.roffasc"): FileFormat.ROFF_ASCII,
        pathlib.Path("wells/battle/1/WELL12.rmswell"): FileFormat.RMSWELL,
    },
)


@pytest.fixture(name="reek_grid_path")
def fixture_reek_grid_path(testpath):
    return pathlib.Path(testpath) / "3dgrids/reek"


def test_fileformat_unknown_empty_memstream():
    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        FileWrapper(io.StringIO()).fileformat()
    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        FileWrapper(io.BytesIO()).fileformat()


@pytest.mark.parametrize("length", [0, 4, 8, 24, 9])
def test_fileformat_unknown_zeroed_memstream_with_varied_length(length):
    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        FileWrapper(io.BytesIO(b"\00" * length)).fileformat()


@pytest.mark.parametrize("filename", FILE_FORMATS.keys())
def test_properties_file(testpath, filename):
    gfile = FileWrapper(testpath / filename)
    assert isinstance(gfile._file, pathlib.Path)

    assert gfile.memstream is False
    assert gfile._mode == "rb"
    assert pathlib.Path(gfile.name) == (testpath / filename).absolute().resolve()

    assert "Swig" in str(gfile.get_cfhandle())
    assert gfile.cfclose() is True


@pytest.mark.parametrize(
    "stream, instance, mode",
    [
        (io.BytesIO(), io.BytesIO, "rb"),
        (io.StringIO(), io.StringIO, "r"),
    ],
)
def test_properties_stream(stream, instance, mode):
    sfile = FileWrapper(stream)
    assert isinstance(sfile._file, instance)
    assert sfile.memstream is True
    assert sfile._mode == mode
    assert sfile.name == stream


@pytest.mark.skipif(sys.platform == "win32", reason="Path delimiter issue")
@pytest.mark.parametrize("filename", SURFACE_FILE_FORMATS.keys())
def test_resolve_alias(testpath, filename):
    """Testing resolving file alias function."""
    surf = xtgeo.surface_from_file(testpath / filename)
    md5hash = surf.generate_hash("md5")

    mname = FileWrapper("whatever/$md5sum.gri", obj=surf)
    assert str(mname.file) == f"whatever/{md5hash}.gri"

    mname = FileWrapper(pathlib.Path("whatever/$md5sum.gri"), obj=surf)
    assert str(mname.file) == f"whatever/{md5hash}.gri"

    mname = FileWrapper("whatever/$random.gri", obj=surf)
    assert len(str(mname.file)) == 45

    # use $fmu.v1 schema
    surf.metadata.opt.shortname = "topValysar"
    surf.metadata.opt.description = "Depth surface"

    mname = FileWrapper(pathlib.Path("whatever/$fmu-v1.gri"), obj=surf)
    assert str(mname.file) == "whatever/topvalysar--depth_surface.gri"


@pytest.mark.parametrize("filename", ["REEK.EGRID", "."])
def test_file_does_exist(reek_grid_path, filename):
    xtgeo_file = FileWrapper(reek_grid_path / filename)
    assert xtgeo_file.exists() is True


@pytest.mark.parametrize("filename", ["NOSUCH.EGRID", "NOSUCH/NOSUCH.EGRID"])
def test_file_does_not_exist(reek_grid_path, filename):
    xtgeo_file = FileWrapper(reek_grid_path / filename)
    assert xtgeo_file.exists() is False


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("REEK.EGRID", True),
        ("REEK.INIT", True),
        ("NOSUCH.EGRID", False),
        ("NOSUCH/NOSUCH.EGRID", False),
    ],
)
def test_check_file(reek_grid_path, filename, expected):
    xtgeo_file = FileWrapper(reek_grid_path / filename)
    assert xtgeo_file.check_file() is expected


@pytest.mark.parametrize("filename", ["NOSUCH.EGRID", "NOSUCH/NOSUCH.EGRID"])
def test_check_file_raises(reek_grid_path, filename):
    xtgeo_file = FileWrapper(reek_grid_path / filename)
    assert xtgeo_file.check_file() is False
    with pytest.raises(OSError):
        xtgeo_file.check_file(raiseerror=OSError)


def test_cannot_reinstance_filewrapper(tmp_path):
    gfile = FileWrapper(tmp_path / "test.txt")
    with pytest.raises(RuntimeError, match="Cannot instantiate"):
        FileWrapper(gfile)


def test_invalid_file_primitive_raises():
    with pytest.raises(RuntimeError, match="Cannot instantiate"):
        FileWrapper(1.0)


def test_resolve_alias_on_stream_doesnt_modify_or_raise():
    stream = io.BytesIO()
    xtg_file = FileWrapper(stream)
    xtg_file.resolve_alias(xtgeo.create_box_grid((1, 1, 1)))
    assert stream == xtg_file.file


def test_bad_alias_raises(tmp_path):
    with pytest.raises(ValueError, match="not a valid alias"):
        FileWrapper(tmp_path / "$NO_ALIAS").resolve_alias(
            xtgeo.create_box_grid((1, 1, 1))
        )


def test_memstream_check_file():
    assert FileWrapper(io.StringIO()).check_file()


@pytest.mark.parametrize("filename", FILE_FORMATS.keys())
def test_file_c_handle(testpath, filename):
    any_xtgeo_file = FileWrapper(testpath / filename)

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


@pytest.mark.bigtest
@pytest.mark.parametrize("filename", SURFACE_FILE_FORMATS.keys())
def test_surface_file_roundtrip_stream(testpath, filename):
    stream = io.BytesIO()
    surf = xtgeo.surface_from_file(testpath / filename)
    surf.to_file(stream)
    stream.seek(0)
    stream_file = FileWrapper(stream)

    assert stream_file.memstream is True
    assert stream_file.fileformat() == FileFormat.IRAP_BINARY


@pytest.mark.parametrize("filename, expected_format", FILE_FORMATS.items())
def test_fileformat_infers_from_suffix(testpath, filename, expected_format):
    xtgeo_file = FileWrapper(testpath / filename)
    assert xtgeo_file.fileformat() == expected_format


@pytest.mark.parametrize("filename, expected_format", FILE_FORMATS.items())
def test_fileformat_infers_from_stream_contents(testpath, filename, expected_format):
    if expected_format in (FileFormat.RMSWELL, FileFormat.ROFF_ASCII):
        with open(testpath / filename) as f:
            stream = io.StringIO(f.read())
    else:
        with open(testpath / filename, "rb") as f:
            stream = io.BytesIO(f.read())
    xtgeo_file = FileWrapper(stream)
    assert xtgeo_file.fileformat() == expected_format


@pytest.mark.parametrize("filename, expected_format", FILE_FORMATS.items())
def test_fileformat_provided(testpath, filename, expected_format):
    xtgeo_file = FileWrapper(testpath / filename)
    name = expected_format.name
    assert xtgeo_file.fileformat(fileformat=name) == expected_format
    assert xtgeo_file.fileformat(fileformat=name.lower()) == expected_format


@pytest.mark.parametrize("filename, expected_format", SURFACE_FILE_FORMATS.items())
def test_fileformat_provided_prefer_given(testpath, filename, expected_format):
    xtgeo_file = FileWrapper(testpath / filename)
    assert xtgeo_file.fileformat(fileformat="segy") == FileFormat.SEGY


@pytest.mark.parametrize("filename", SURFACE_FILE_FORMATS.keys())
def test_fileformat_hdf_stream(testpath, filename):
    stream = io.BytesIO()
    surf = xtgeo.surface_from_file(testpath / filename)
    surf.to_hdf(stream)
    stream.seek(0)
    sfile = FileWrapper(stream)
    assert sfile.memstream is True
    assert sfile.fileformat() == FileFormat.HDF


@pytest.mark.parametrize("filename", SURFACE_FILE_FORMATS.keys())
def test_fileformat_hdf_to_file(tmp_path, testpath, filename):
    newfile = tmp_path / "hdf_surf.hdf"
    surf = xtgeo.surface_from_file(testpath / filename)
    surf.to_hdf(newfile)
    sfile = FileWrapper(newfile)
    assert sfile.fileformat() == FileFormat.HDF
