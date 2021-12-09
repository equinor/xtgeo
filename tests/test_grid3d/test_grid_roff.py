import io
import re
from collections import OrderedDict
from itertools import product

import hypothesis.strategies as st
import numpy as np
import pytest
import roffio
from hypothesis import given
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.grid3d import Grid
from xtgeo.grid3d._grid_import_roff import handle_deprecated_xtgeo_roff_file
from xtgeo.grid3d._roff_grid import RoffGrid

from .grid_generator import dimensions, xtgeo_grids

finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


def local_to_utm(roff_grid, coordinates):
    (x, y, z) = coordinates
    x_utm = (x + roff_grid.xoffset) * roff_grid.xscale
    y_utm = (y + roff_grid.yoffset) * roff_grid.yscale
    tvd = (z + roff_grid.zoffset) * roff_grid.zscale
    return (x_utm, y_utm, tvd)


def line_vertices(roff_grid, i, j):
    pos = 6 * (i * (roff_grid.ny + 1) + j)
    x_bot = roff_grid.corner_lines[pos]
    y_bot = roff_grid.corner_lines[pos + 1]
    z_bot = roff_grid.corner_lines[pos + 2]
    x_top = roff_grid.corner_lines[pos + 3]
    y_top = roff_grid.corner_lines[pos + 4]
    z_top = roff_grid.corner_lines[pos + 5]

    return ((x_bot, y_bot, z_bot), (x_top, y_top, z_top))


def points(roff_grid):
    return product(
        range(1, roff_grid.nx - 1),
        range(1, roff_grid.ny - 1),
        range(1, roff_grid.nz - 1),
    )


def layer_points(roff_grid):
    return product(range(1, roff_grid.nx - 1), range(1, roff_grid.ny - 1))


def same_geometry(roff_grid, other_roff_grid):
    if not isinstance(other_roff_grid, RoffGrid):
        return False

    is_same = True
    for line in layer_points(roff_grid):
        for v1, v2 in zip(
            line_vertices(roff_grid, *line), line_vertices(other_roff_grid, *line)
        ):
            is_same = is_same and np.allclose(
                local_to_utm(roff_grid, v1), local_to_utm(other_roff_grid, v2), atol=0.1
            )

    for node in points(roff_grid):
        is_same = is_same and np.allclose(
            (other_roff_grid.z_value(node) + other_roff_grid.zoffset)
            * other_roff_grid.zscale,
            (roff_grid.z_value(node) + roff_grid.zoffset) * roff_grid.zscale,
            atol=0.2,
        )

    return is_same


@st.composite
def subgrids(draw, nz):
    if draw(st.booleans()):
        return None

    result = []
    res_sum = 0
    while res_sum < nz:
        sublayers = draw(st.integers(min_value=1, max_value=nz - res_sum))
        result.append(sublayers)
        res_sum += sublayers
    return np.array(result, dtype=np.int32)


@st.composite
def roff_grids(draw, dim=dimensions):
    dims = draw(dim)
    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    corner_lines = draw(
        arrays(
            shape=corner_size,
            dtype=np.float32,
            elements=finites,
        )
    )
    num_nodes = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)
    split_enz = draw(
        arrays(shape=num_nodes, dtype=np.int8, elements=st.sampled_from([1, 4]))
    ).tobytes()
    if split_enz is not None:
        numz = sum(split_enz)
    else:
        numz = num_nodes
    zvals = draw(arrays(shape=int(numz), dtype=np.float32, elements=finites))
    active = draw(
        arrays(
            shape=dims[0] * dims[1] * dims[2], dtype=np.bool_, elements=st.just(True)
        )
    )

    subs = draw(subgrids(dims[2]))

    rest = draw(st.tuples(*([finites] * 6)))
    return RoffGrid(*dims, corner_lines, zvals, split_enz, active, subs, *rest)


@given(roff_grids())
def test_roff_grid_read_write(rgrid):
    buff = io.BytesIO()
    rgrid.to_file(buff)

    buff.seek(0)
    assert RoffGrid.from_file(buff) == rgrid


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format2(xtggrid):
    xtggrid._xtgformat2()
    roff_grid = RoffGrid.from_xtgeo_grid(xtggrid)

    assert_allclose(roff_grid.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)
    assert roff_grid.xtgeo_subgrids() == xtggrid._subgrids


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format1(xtggrid):
    xtggrid._xtgformat1()
    roff_grid = RoffGrid.from_xtgeo_grid(xtggrid)

    xtggrid._xtgformat2()
    assert_allclose(roff_grid.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)
    assert roff_grid.xtgeo_subgrids() == xtggrid._subgrids


@given(roff_grids())
def test_to_from_roffgrid(roff_grid):
    xtggrid = Grid(
        actnumsv=roff_grid.xtgeo_actnum(),
        coordsv=roff_grid.xtgeo_coord(),
        zcornsv=roff_grid.xtgeo_zcorn(),
        subgrids=roff_grid.xtgeo_subgrids(),
    )

    roffgrid2 = RoffGrid.from_xtgeo_grid(xtggrid)
    assert same_geometry(roffgrid2, roff_grid)
    assert np.array_equal(roffgrid2.subgrids, roff_grid.subgrids)


def test_missing_non_optionals():
    buff = io.BytesIO()
    roffio.write(buff, {"filedata": {"filetype": "grid"}})
    buff.seek(0)
    with pytest.raises(ValueError, match="Missing non-optional"):
        RoffGrid.from_file(buff)


@given(roff_grids())
def test_not_a_grid(roff_grid):
    buff = io.BytesIO()
    roff_grid.to_file(buff)
    buff.seek(0)
    values = roffio.read(buff)
    values["filedata"]["filetype"] = "notgrid"
    buff.seek(0)
    roffio.write(buff, values)
    buff.seek(0)
    with pytest.raises(ValueError, match="did not have filetype set to grid"):
        RoffGrid.from_file(buff)


# pylint: disable=redefined-outer-name
@pytest.fixture
def single_cell_roff_grid():
    corner_lines = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    ).ravel()
    splitenz = np.ones(8, dtype=np.uint8).tobytes()
    active = np.ones(1, dtype=bool)
    zvals = np.ones(1 * len(splitenz), dtype=np.float32)
    return RoffGrid(1, 1, 1, corner_lines, zvals, splitenz, active)


def test_unsupported_split_enz(single_cell_roff_grid):
    roff_grid = single_cell_roff_grid
    roff_grid.split_enz = np.full(8, fill_value=2, dtype=np.uint8).tobytes()
    roff_grid.zvals = np.ones(16, dtype=np.float32)

    with pytest.raises(ValueError, match="split type"):
        roff_grid.xtgeo_zcorn()


def test_too_few_zvals(single_cell_roff_grid):
    roff_grid = single_cell_roff_grid
    roff_grid.split_enz = np.full(8, fill_value=4, dtype=np.uint8).tobytes()
    roff_grid.zvals = np.ones(5, dtype=np.float32)

    with pytest.raises(ValueError, match="size of zdata"):
        roff_grid.xtgeo_zcorn()


def test_too_many_zvals(single_cell_roff_grid):
    roff_grid = single_cell_roff_grid
    roff_grid.split_enz = np.full(8, fill_value=4, dtype=np.uint8).tobytes()
    roff_grid.zvals = np.ones(300, dtype=np.float32)

    with pytest.raises(ValueError, match="size of zdata"):
        roff_grid.xtgeo_zcorn()


def test_wrong_zcornsv_size():
    split_enz = np.full(8, fill_value=4, dtype=np.uint8).tobytes()
    zvals = np.ones(300, dtype=np.float32)
    zcornsv = np.zeros(5, dtype=np.float32)
    retval = _cxtgeo.grd3d_roff2xtgeo_splitenz(3, 1.0, 1.0, split_enz, zvals, zcornsv)
    assert retval == -4


@given(roff_grids())
def test_xtgeo_values_are_c_contiguous(roff_grid):
    assert roff_grid.xtgeo_coord().flags["C_CONTIGUOUS"]
    assert roff_grid.xtgeo_actnum().flags["C_CONTIGUOUS"]
    assert roff_grid.xtgeo_zcorn().flags["C_CONTIGUOUS"]


@given(roff_grids())
def test_default_values(roff_grid):
    buff = io.BytesIO()
    roff_grid.to_file(buff)
    buff.seek(0)
    values = roffio.read(buff)

    del values["translate"]
    del values["scale"]
    if "subgrids" in values:
        del values["subgrids"]
    del values["active"]

    buff2 = io.BytesIO()
    roffio.write(buff2, values)
    buff2.seek(0)
    roff_grid2 = RoffGrid.from_file(buff2)

    assert roff_grid2.xoffset == 0.0
    assert roff_grid2.yoffset == 0.0
    assert roff_grid2.zoffset == 0.0

    assert roff_grid2.xscale == 1.0
    assert roff_grid2.yscale == 1.0
    assert roff_grid2.zscale == -1.0

    assert roff_grid2.subgrids is None

    assert np.array_equal(
        roff_grid2.active,
        np.ones(roff_grid.nx * roff_grid.ny * roff_grid.nz, dtype=np.bool_),
    )


@given(roff_grids())
def test_eq_reflexivity(roff_grid):
    assert roff_grid == roff_grid


@given(roff_grids(), roff_grids())
def test_eq_symmetry(roff_grid1, roff_grid2):
    if roff_grid1 == roff_grid2:
        assert roff_grid2 == roff_grid1


@given(roff_grids(), roff_grids(), roff_grids())
def test_eq_transitivity(roff_grid1, roff_grid2, roff_grid3):
    if roff_grid1 == roff_grid2 and roff_grid2 == roff_grid3:
        assert roff_grid1 == roff_grid3


def test_from_xtgeo_subgrids():
    assert list(RoffGrid._from_xtgeo_subgrids(OrderedDict())) == []
    assert list(
        RoffGrid._from_xtgeo_subgrids(OrderedDict([("subgrid_0", range(1, 2))]))
    ) == [1]
    assert list(RoffGrid._from_xtgeo_subgrids(OrderedDict([("subgrid_0", [1])]))) == [1]
    assert list(
        RoffGrid._from_xtgeo_subgrids(
            OrderedDict([("subgrid_0", [1, 2, 3]), ("subgrid_1", [4])])
        )
    ) == [3, 1]
    assert list(
        RoffGrid._from_xtgeo_subgrids(
            OrderedDict([("subgrid_0", range(1, 4)), ("subgrid_1", range(4, 5))])
        )
    ) == [3, 1]


XTGEO_214_HEADER = (
    b"roff-bin\0"
    b"#ROFF file#\0"
    b"#Creator: CXTGeo subsystem of XTGeo by JCR#\0"
    b"tag\0filedata\0"
    b"int\0byteswaptest\0"
    + (1).to_bytes(4, byteorder="little")
    + b"char\0filetype\0grid\0char\0creationDate\0UNKNOWN"
)


@given(roff_grids(dim=st.tuples(*([st.integers(min_value=4, max_value=5)] * 3))))
def test_deprecated_fileread(roff_grid):
    buff = io.BytesIO()
    roff_grid.to_file(buff)

    new_buff = io.BytesIO(
        re.sub(
            b".*creationDate\0[^\0]*\0char\0filetype\0grid\0",
            XTGEO_214_HEADER,
            buff.getvalue(),
        )
    )

    with pytest.warns(UserWarning, match="nonstandard but harmless roff"):
        with handle_deprecated_xtgeo_roff_file(new_buff) as converted_buff:
            new_grid = RoffGrid.from_file(converted_buff)

    assert new_grid == roff_grid
