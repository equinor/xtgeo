import hypothesis.strategies as st
import numpy as np
import xtgeo.grid3d._ecl_grid as eclgrid
import xtgeo.grid3d._grdecl_grid as ggrid
from hypothesis.extra.numpy import arrays

from .grid_generator import indices

finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)

units = st.sampled_from(eclgrid.Units)
grid_relatives = st.sampled_from(eclgrid.GridRelative)
orders = st.sampled_from(eclgrid.Order)
orientations = st.sampled_from(eclgrid.Orientation)
handedness = st.sampled_from(eclgrid.Handedness)
coordinate_types = st.sampled_from(eclgrid.CoordinateType)

map_axes = st.builds(
    ggrid.MapAxes,
    st.tuples(finites, finites),
    st.tuples(finites, finites),
    st.tuples(finites, finites),
).filter(ggrid.GrdeclGrid.valid_mapaxes)

gdorients = st.builds(ggrid.GdOrient, orders, orders, orders, orientations, handedness)


@st.composite
def gridunits(draw, relative=grid_relatives):
    return draw(st.builds(ggrid.GridUnit, units, relative))


@st.composite
def specgrids(
    draw, coordinates=coordinate_types, numres=st.integers(min_value=1, max_value=3)
):
    return draw(
        st.builds(
            ggrid.SpecGrid,
            indices,
            indices,
            indices,
            numres,
            coordinates,
        )
    )


@st.composite
def zcorns(draw, dims):
    return draw(
        arrays(
            shape=8 * dims[0] * dims[1] * dims[2],
            dtype=np.float32,
            elements=finites,
        )
    )


@st.composite
def grdecl_grids(
    draw,
    spec=specgrids(),
    mpaxs=map_axes,
    orient=gdorients,
    gunit=gridunits(),
    zcorn=zcorns,
):
    specgrid = draw(spec)
    dims = specgrid.ndivix, specgrid.ndiviy, specgrid.ndiviz

    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    coord = draw(
        arrays(
            shape=corner_size,
            dtype=np.float32,
            elements=finites,
        )
    )
    if draw(st.booleans()):
        actnum = draw(
            arrays(
                shape=dims[0] * dims[1] * dims[2],
                dtype=np.int32,
                elements=st.integers(min_value=0, max_value=3),
            )
        )
    else:
        actnum = None
    mapax = draw(mpaxs) if draw(st.booleans()) else None
    gdorient = draw(orient) if draw(st.booleans()) else None
    gridunit = draw(gunit) if draw(st.booleans()) else None

    return ggrid.GrdeclGrid(
        mapaxes=mapax,
        specgrid=specgrid,
        gridunit=gridunit,
        zcorn=draw(zcorn(dims)),
        actnum=actnum,
        coord=coord,
        gdorient=gdorient,
    )


@st.composite
def xtgeo_compatible_zcorns(draw, dims):
    nx, ny, nz = dims
    array = draw(
        arrays(
            shape=(2, nx, 2, ny, 2, nz),
            dtype=np.float32,
            elements=finites,
        )
    )
    array[:, :, :, :, 1, : nz - 1] = array[:, :, :, :, 0, 1:]
    return array.ravel(order="F")


xtgeo_compatible_grdecl_grids = grdecl_grids(
    spec=specgrids(
        coordinates=st.just(ggrid.CoordinateType.CARTESIAN),
        numres=st.just(1),
    ),
    orient=st.just(ggrid.GdOrient()),
    gunit=gridunits(relative=st.just(ggrid.GridRelative.ORIGIN)),
    zcorn=xtgeo_compatible_zcorns,
)
