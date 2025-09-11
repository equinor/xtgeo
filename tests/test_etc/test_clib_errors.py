import io

import numpy as np
import pytest

import xtgeo
from xtgeo import _cxtgeo
from xtgeo.io._file import FileWrapper


@pytest.mark.parametrize("func", [_cxtgeo.x_ic2ijk, _cxtgeo.x_ib2ijk])
def test_calc_i_to_ijk(func):
    with pytest.raises(
        xtgeo.XTGeoCLibError, match=f"Critical error in: {func.__name__}"
    ):
        func(0, 3, 4, 5, 2)


def test_surf_export_petromod_exception_no_file():
    with pytest.raises(
        xtgeo.XTGeoCLibError, match="Cannot open file in: surf_export_petromod_bin"
    ):
        _cxtgeo.surf_export_petromod_bin(
            None,
            "not_relevant",
            [1],
        )


def test_surf_export_petromod_exception():
    gfile = FileWrapper(io.BytesIO(b"\x00"))
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Error writing to Storm format. Bug in: surf_export_petromod_bi",
    ):
        _cxtgeo.surf_export_petromod_bin(
            gfile.get_cfhandle(),
            "not_relevant",
            [1, 2],
        )


# @pytest.mark.xfail(reason="Quite hard to make test case")
# def test_grd3d_ecl_tsteps():
#     gfile = FileWrapper(io.BytesIO(b"\x00"))
#     seq = _cxtgeo.new_intarray(10)
#     day = _cxtgeo.new_intarray(10)
#     mon = _cxtgeo.new_intarray(10)
#     yer = _cxtgeo.new_intarray(10)
#
#     with pytest.raises(
#         xtgeo.XTGeoCLibError,
#         match="Fail in dimensions in ",
#     ):
#         _cxtgeo.grd3d_ecl_tsteps(
#             gfile.get_cfhandle(),
#             seq,
#             day,
#             mon,
#             yer,
#             10,
#         )


@pytest.mark.parametrize(
    "bytestring, mx, expected_msg",
    [
        (b"\x00", 1, r"mx \* my != nsurf"),
        (b"\x00\x00", 2, "Failed to read file in: surf_import_petromod_bin"),
        (b"\x00\x00\x00\x00", 2, "Error when reading file in:"),
    ],
)
def test_surf_import_petromod_bin(bytestring, mx, expected_msg):
    gfile = FileWrapper(io.BytesIO((bytestring)))
    with pytest.raises(xtgeo.XTGeoCLibError, match=expected_msg):
        _cxtgeo.surf_import_petromod_bin(gfile.get_cfhandle(), 1, 0.0, mx, 2, 4)


def test_surf_sample_grd3d_lay():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Errors in array lengths checks in:",
    ):
        _cxtgeo.surf_sample_grd3d_lay(
            1,
            1,
            1,
            np.array([0.0]),
            np.array([1.0]),
            np.array([1], dtype=np.int32),
            1,
            1,
            1,
            1,
            1.0,
            1.0,
            1.0,
            1.0,
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            1,
        )


def test_grd3d_reduce_onelayer():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="IFLAG other than 0 not implemented",
    ):
        _cxtgeo.grd3d_reduce_onelayer(
            0,
            0,
            0,
            np.array([0.0]),
            np.array([1.0]),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            _cxtgeo.new_intarray(1),
            1,
        )


def test_grd3cp3d_xtgformat1to2_geom():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Error in: grd3cp3d_xtgformat1to2_geom, ib != nzcorn2",
    ):
        _cxtgeo.grd3cp3d_xtgformat1to2_geom(
            -1,
            -1,
            -1,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )


def test_grd3cp3d_xtgformat2to1_geom():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Error in grd3cp3d_xtgformat2to1_geom, ib != nzcorn1",
    ):
        _cxtgeo.grd3cp3d_xtgformat2to1_geom(
            -1,
            -1,
            -1,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
