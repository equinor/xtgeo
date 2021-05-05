import pytest
import io
import numpy as np
import stat
import os
import xtgeo
from xtgeo.cxtgeo import _cxtgeo


@pytest.mark.parametrize(
    "bytevalue, array_length, expected_result",
    [
        (b"\x01", 1, [0, 0]),
        (b"\x00\x00\x01\x01", 2, [257, 0]),
    ],
)
def test_grd3d_imp_roffbin_ilist_exception(bytevalue, array_length, expected_result):
    gfile = xtgeo._XTGeoFile(io.BytesIO(bytevalue))
    inumpy = np.zeros(2, dtype=np.int32)
    with pytest.raises(xtgeo.XTGeoCLibError, match="Error reading file"):
        _cxtgeo.grd3d_imp_roffbin_ilist(gfile.get_cfhandle(), 1, 0, inumpy)
    assert list(inumpy) == expected_result


@pytest.fixture
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


@pytest.mark.usefixtures("setup_tmpdir")
def test_eclgrid_no_file_access():
    fname = "random_file_name"
    with open(fname, "w"):
        pass
    os.chmod(fname, stat.S_IREAD)
    with pytest.raises(xtgeo.XTGeoCLibError, match=f"Could not open file: {fname}"):
        # The input here is not very important, what is important
        # is that "existing_file" can not be opened.
        _cxtgeo.grd3d_export_egrid(
            1,
            1,
            1,
            [1],
            [1],
            np.array([1], dtype=np.int32),
            fname,
            0,
        )
    os.chmod(fname, stat.S_IWRITE)
    os.remove(fname)


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
        match="Bug in grdcp3d_from_cube",
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
        xtgeo.XTGeoCLibError, match=f"Critical error from {func.__name__}"
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


@pytest.mark.parametrize("swap", [0, 1, 2])
@pytest.mark.parametrize("nsplit", [4, 1])
def test_grdcp3d_imp_roffbin_zcornsv_swap_0(nsplit, swap):
    gfile = xtgeo._XTGeoFile(io.BytesIO(b"\x00\x00"))
    int_array = _cxtgeo.new_intarray(8)
    _cxtgeo.intarray_setitem(int_array, 0, nsplit)
    with pytest.raises(
        xtgeo.XTGeoCLibError,
        match=f"Failed to read file, swap: {swap}, for nsplit: {nsplit}, n: 0",
    ):
        _cxtgeo.grdcp3d_imp_roffbin_zcornsv(
            gfile.get_cfhandle(),
            swap,
            0,
            2,
            2,
            2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            int_array,
            1,
            np.array(0.0, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "func, vector",
    [
        (_cxtgeo.grd3d_imp_roffbin_ivec, _cxtgeo.new_intarray(1)),
        (_cxtgeo.grd3d_imp_roffbin_fvec, _cxtgeo.new_floatarray(1)),
        (_cxtgeo.grd3d_imp_roffbin_bvec, _cxtgeo.new_intarray(1)),
    ],
)
def test_grd3d_imp_roffbin_ivec(func, vector):
    gfile = xtgeo._XTGeoFile(io.BytesIO(b""))
    with pytest.raises(xtgeo.XTGeoCLibError, match="Failed to read from file"):
        func(gfile.get_cfhandle(), 1, 0, vector, 1)
