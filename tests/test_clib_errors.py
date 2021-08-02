import io
import os
import stat

import numpy as np
import pytest

import xtgeo
from xtgeo.cxtgeo import _cxtgeo


@pytest.fixture()
def unreadable_file(setup_tmpdir):
    fname = "random_file_name"
    with open(fname, "w"):
        pass
    os.chmod(fname, stat.S_IREAD)
    # On some systems the chmod fails, meaning we are able to write to the
    # file. In those cases we skip the test:
    if os.access(fname, os.W_OK):
        pytest.skip("Have write access to file")
    yield fname
    os.chmod(fname, stat.S_IWRITE)
    os.remove(fname)


def test_eclgrid_no_file_access(unreadable_file):
    with pytest.raises(xtgeo.XTGeoCLibError, match="Could not open file in"):
        # The input here is not very important, what is important
        # is that "existing_file" can not be opened.
        _cxtgeo.grd3d_export_egrid(
            1,
            1,
            1,
            [1],
            [1],
            np.array([1], dtype=np.int32),
            unreadable_file,
            0,
        )


def test_grd3d_calc_dxdy():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Errors in array lengths checks in grd3d_calc_dxdy",
    ):
        _cxtgeo.grd3d_calc_dxdy(
            1,
            1,
            1,
            np.array([0.0]),
            np.array([1.0]),
            np.array([1], dtype=np.int32),
            np.array([0.0]),
            np.array([0.0]),
            0,
            0,
        )


def test_grd3d_get_xyz():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Errors in array lengths checks in grd3d_calc_xyz",
    ):
        _cxtgeo.grd3d_calc_xyz(
            1,
            1,
            1,
            np.array([0.0]),
            np.array([1.0]),
            np.array([1], dtype=np.int32),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            0,
        )


def test_grdcp3d_from_cube():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Bug in: grdcp3d_from_cube",
    ):
        _cxtgeo.grdcp3d_from_cube(
            1,
            1,
            1,
            np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
                dtype=np.float32,
            ),
            np.array(
                [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], dtype=np.int32
            ),
            1.0,
            1.0,
            1.0,
            -1,
            1,
            1,
            0.0,
            0,
            1,
        )


@pytest.mark.parametrize("func", [_cxtgeo.x_ic2ijk, _cxtgeo.x_ib2ijk])
def test_calc_i_to_ijk(func):
    with pytest.raises(
        xtgeo.XTGeoCLibError, match=f"Critical error in: {func.__name__}"
    ):
        func(0, 3, 4, 5, 2)


def test_export_grid_cornerpoint_roxapi_v1():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Errors in array lengths checks in grd3d_conv_grid_roxapi",
    ):
        _cxtgeo.grd3d_conv_grid_roxapi(
            1,
            1,
            1,
            np.array([0.0]),
            np.array([1.0]),
            np.array([1], dtype=np.int32),
            1,
            1,
            1,
        )


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
    gfile = xtgeo._XTGeoFile(io.BytesIO(b"\x00"))
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
#     gfile = xtgeo._XTGeoFile(io.BytesIO(b"\x00"))
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
    gfile = xtgeo._XTGeoFile(io.BytesIO((bytestring)))
    with pytest.raises(xtgeo.XTGeoCLibError, match=expected_msg):
        _cxtgeo.surf_import_petromod_bin(gfile.get_cfhandle(), 1, 0.0, mx, 2, 4)


def test_grd3d_export_grdeclprop_no_file(unreadable_file):
    with pytest.raises(xtgeo.XTGeoCLibError, match="Could not open file"):
        # The input here is not very important, what is important
        # is that "existing_file" can not be opened.
        _cxtgeo.grd3d_export_grdeclprop2(
            1,
            1,
            1,
            1,
            _cxtgeo.new_intpointer(),
            _cxtgeo.new_floatpointer(),
            _cxtgeo.new_doublepointer(),
            "name",
            " %d",
            unreadable_file,
            1,
            0,
        )


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


def test_grd3d_read_eclrecord():
    with pytest.raises(xtgeo.XTGeoCLibError, match="Cannot use file"):
        _cxtgeo.grd3d_read_eclrecord(
            None,
            1,
            1,
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.float32),
            np.array([1], dtype=np.float64),
        )


def test_grd3d_from_cube():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Bug in grd3d_from_cube",
    ):
        _cxtgeo.grd3d_from_cube(
            2,
            2,
            1,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),  # not relevant
            np.array([1, 1, 1, 1], dtype=np.int32),  # not relevant
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            0,
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


def test_grd3d_imp_ecl_egrid():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match="Errors in array lengths checks in",
    ):
        _cxtgeo.grd3d_imp_ecl_egrid(
            xtgeo._XTGeoFile(io.BytesIO(b"")).get_cfhandle(),
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1], dtype=np.int32),
            _cxtgeo.new_longarray(1),
            1,
        )


def test_grd3d_points_ijk_cells_nxvec():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match=r"nxvec != nyvec or nyvec != nzvec",
    ):
        carr = [_cxtgeo.new_doublearray(1) for _ in range(4)]
        [_cxtgeo.swig_numpy_to_carr_1d(np.array([1.0]), arr) for arr in carr]

        _cxtgeo.grd3d_points_ijk_cells(
            np.array([1.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            carr[0],
            carr[1],
            carr[2],
            carr[3],
            1,
            1,
            1,
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.0], dtype=np.float64),
            1,
            1,
            1,
            1,
        )


def test_grd3d_points_ijk_cells_nivec():
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match=r"nivec != njvec or nivec != nkvec",
    ):
        carr = [_cxtgeo.new_doublearray(1) for _ in range(4)]
        [_cxtgeo.swig_numpy_to_carr_1d(np.array([1.0]), arr) for arr in carr]

        _cxtgeo.grd3d_points_ijk_cells(
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1,
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            carr[0],
            carr[1],
            carr[2],
            carr[3],
            1,
            1,
            1,
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.0], dtype=np.float64),
            1,
            2,
            1,
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
