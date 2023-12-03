import hypothesis.strategies as st
import numpy as np
import xtgeo.grid3d._egrid as xtge
from hypothesis.extra.numpy import arrays

from .grdecl_grid_generator import (
    coordinate_types,
    finites,
    gdorients,
    gridunits,
    map_axes,
    units,
    xtgeo_compatible_zcorns,
    zcorns,
)
from .grid_generator import indices


@st.composite
def ascii_string(draw, min_size=0, max_size=8):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    result = b""
    for _ in range(size):
        # Printable non-extended ascii characters are between 32 and 126
        result += draw(st.integers(min_value=32, max_value=126)).to_bytes(1, "little")
    return result.decode("ascii")


types_of_grid = st.just(xtge.TypeOfGrid.CORNER_POINT)
rock_models = st.sampled_from(xtge.RockModel)
grid_formats = st.just(xtge.GridFormat.IRREGULAR_CORNER_POINT)
file_heads = st.builds(
    xtge.Filehead,
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=2000, max_value=2022),
    st.integers(min_value=0, max_value=5),
    types_of_grid,
    rock_models,
    grid_formats,
)


@st.composite
def grid_heads(
    draw,
    gridtype=types_of_grid,
    nx=indices,
    ny=indices,
    nz=indices,
    index=st.integers(min_value=0, max_value=5),
    coordinatesystem=coordinate_types,
):
    return xtge.GridHead(
        draw(gridtype),
        draw(nx),
        draw(ny),
        draw(nz),
        draw(index),
        1,
        1,
        draw(coordinatesystem),
        draw(st.tuples(indices, indices, indices)),
        draw(st.tuples(indices, indices, indices)),
    )


@st.composite
def global_grids(draw, header=grid_heads(), zcorn=zcorns):
    grid_head = draw(header)
    dims = (grid_head.num_x, grid_head.num_y, grid_head.num_z)
    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    coord = arrays(
        shape=corner_size,
        dtype=np.float32,
        elements=finites,
    )
    actnum = st.one_of(
        st.just(None),
        arrays(
            shape=dims[0] * dims[1] * dims[2],
            dtype=np.int32,
            elements=st.integers(min_value=0, max_value=3),
        ),
    )
    return xtge.GlobalGrid(
        coord=draw(coord),
        zcorn=draw(zcorn(dims)),
        actnum=draw(actnum),
        grid_head=grid_head,
        coord_sys=draw(map_axes),
        boxorig=draw(st.tuples(indices, indices, indices)),
        corsnum=draw(
            arrays(elements=indices, dtype="int32", shape=indices),
        ),
    )


@st.composite
def lgr_sections(draw, nx=st.just(2), ny=st.just(2), nz=st.just(2), zcorn=zcorns):
    grid_head = draw(grid_heads(nx=nx, ny=ny, nz=nz))
    dims = (grid_head.num_x, grid_head.num_y, grid_head.num_z)
    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    coord = arrays(
        shape=corner_size,
        dtype=np.float32,
        elements=finites,
    )
    actnum = st.one_of(
        st.just(None),
        arrays(
            shape=dims[0] * dims[1] * dims[2],
            dtype=np.int32,
            elements=st.integers(min_value=0, max_value=3),
        ),
    )
    return xtge.LGRSection(
        coord=draw(coord),
        zcorn=draw(zcorn(dims)),
        actnum=draw(actnum),
        grid_head=grid_head,
        name=draw(ascii_string(min_size=1)),
        parent=draw(st.one_of(st.just(None), ascii_string(min_size=1))),
        grid_parent=draw(st.one_of(st.just(None), ascii_string(min_size=1))),
        hostnum=draw(arrays(elements=indices, dtype="int32", shape=indices)),
        boxorig=draw(st.tuples(indices, indices, indices)),
        coord_sys=draw(map_axes),
    )


nnc_heads = st.builds(xtge.NNCHead, indices, indices)

nnc_sections = st.one_of(
    st.builds(
        xtge.NNCSection,
        nnc_heads,
        arrays(elements=indices, dtype="int32", shape=st.just(2)),
        arrays(elements=indices, dtype="int32", shape=st.just(2)),
        arrays(elements=indices, dtype="int32", shape=st.just(2)),
        arrays(elements=indices, dtype="int32", shape=st.just(2)),
    ),
    st.builds(
        xtge.AmalgamationSection,
        st.tuples(indices, indices),
        arrays(elements=indices, dtype="int32", shape=st.just(2)),
        arrays(elements=indices, dtype="int32", shape=st.just(2)),
    ),
)


@st.composite
def egrid_heads(draw, mpaxes=map_axes):
    return xtge.EGridHead(
        draw(file_heads),
        draw(units),
        draw(mpaxes),
        draw(gridunits()),
        draw(gdorients),
    )


@st.composite
def egrids(
    draw,
    head=egrid_heads(),
    global_grid=global_grids(),
    lgrs=st.lists(lgr_sections(), max_size=2),
    nncs=st.lists(nnc_sections, max_size=2),
):
    return xtge.EGrid(draw(head), draw(global_grid), draw(lgrs), draw(nncs))


xtgeo_compatible_global_grids = global_grids(
    header=grid_heads(coordinatesystem=st.just(xtge.CoordinateType.CARTESIAN)),
    zcorn=xtgeo_compatible_zcorns,
)


@st.composite
def xtgeo_compatible_egridheads(draw, grdunit=gridunits(), mpaxes=map_axes):
    return xtge.EGridHead(
        xtge.Filehead(
            3,
            2007,
            3,
            xtge.TypeOfGrid.CORNER_POINT,
            xtge.RockModel.SINGLE_PERMEABILITY_POROSITY,
            xtge.GridFormat.IRREGULAR_CORNER_POINT,
        ),
        gridunit=draw(grdunit),
        mapaxes=draw(mpaxes),
    )


def xtgeo_compatible_egrids(
    head=xtgeo_compatible_egridheads(),
    global_grid=xtgeo_compatible_global_grids,
    lgrs=st.just([]),
    nncs=st.just([]),
):
    return egrids(head, global_grid, lgrs, nncs)
