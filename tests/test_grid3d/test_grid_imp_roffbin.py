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


def test_eclgrid_no_file_access(tmpdir):
    with tmpdir.as_cwd():
        with open("existing_file", "w"):
            pass
        os.chmod("existing_file", stat.S_IREAD)
        with pytest.raises(
            xtgeo.XTGeoCLibError, match="Could not open file: existing_file"
        ):
            # The input here is not very important, what is important
            # is that "existing_file" can not be opened.
            _cxtgeo.grd3d_export_egrid(
                1,
                1,
                1,
                [1],
                [1],
                np.array([1], dtype=np.int32),
                "existing_file",
                0,
            )
        os.chmod("existing_file", stat.S_IWRITE)


def test_grd3d_calc_dxdy(tmpdir):
    with tmpdir.as_cwd():
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


@pytest.mark.parametrize("func", [_cxtgeo.x_ic2ijk, _cxtgeo.x_ib2ijk])
def test_calc_i_to_ijk(func):
    with pytest.raises(
        xtgeo.XTGeoCLibError, match=f"Critical error from {func.__name__}"
    ):
        func(0, 3, 4, 5, 2)
