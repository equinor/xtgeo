# coding: utf-8
"""Testing: test_grid_property"""


import os
import pathlib

import hypothesis.strategies as st
import numpy as np
import numpy.ma as npma
import pytest
from hypothesis import HealthCheck, example, given, settings

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common.exceptions import KeywordNotFoundError
from xtgeo.grid3d import Grid, GridProperty
from xtgeo.xyz import Polygons

from .grid_generator import dimensions, xtgeo_grids

# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name

# set default level
xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

TESTFILE1 = TPATH / "3dgrids/reek/reek_sim_poro.roff"
TESTFILE2 = TPATH / "3dgrids/eme/1/emerald_hetero.roff"
TESTFILE5 = TPATH / "3dgrids/reek/REEK.EGRID"
TESTFILE6 = TPATH / "3dgrids/reek/REEK.INIT"
TESTFILE7 = TPATH / "3dgrids/reek/REEK.UNRST"
TESTFILE8 = TPATH / "3dgrids/reek/reek_sim_zone.roff"
TESTFILE8A = TPATH / "3dgrids/reek/reek_sim_grid.roff"
TESTFILE9 = TESTFILE1
TESTFILE10 = TPATH / "3dgrids/bri/b_grid.roff"
TESTFILE11 = TPATH / "3dgrids/bri/b_poro.roff"
POLYFILE = TPATH / "polygons/reek/1/polset2.pol"

TESTFILE12A = TPATH / "3dgrids/reek/reek_sim_grid.grdecl"
TESTFILE12B = TPATH / "3dgrids/reek/reek_sim_poro.grdecl"

TESTFILE13A = TPATH / "3dgrids/etc/TEST_SP.EGRID"
TESTFILE13B = TPATH / "3dgrids/etc/TEST_SP.INIT"

DUALROFF = TPATH / "3dgrids/etc/dual_grid_w_props.roff"

BANAL7 = TPATH / "3dgrids/etc/banal7_grid_params.roff"


def test_create():
    """Create a simple property"""

    x = GridProperty()
    assert x.ncol == 4, "NCOL"
    assert x.nrow == 3, "NROW"

    m = GridProperty(discrete=True)
    assert m.isdiscrete


def test_banal7(show_plot):
    """Create a simple property in a small grid box"""

    grd = xtgeo.grid_from_file(BANAL7)
    assert grd.dimensions == (4, 2, 3)
    disc = xtgeo.gridproperty_from_file(BANAL7, name="DISC")
    assert disc.dimensions == (4, 2, 3)
    assert disc.values.mean() == pytest.approx(0.59091, abs=0.001)

    gprops = grd.get_gridquality_properties()
    mix = gprops.get_prop_by_name("minangle_sides")
    assert mix.values.mean() == pytest.approx(81.31036, abs=0.001)

    if show_plot:
        lay = 2
        layslice = xtgeo.plot.Grid3DSlice()
        layslice.canvas(title=f"Layer {lay}")
        layslice.plot_gridslice(
            grd,
            prop=mix,
            mode="layer",
            index=lay,
            window=None,
            linecolor="black",
        )

        layslice.show()


def test_assign():
    """Create a simple property and assign all values a constant"""

    vals = np.array(range(12)).reshape((3, 2, 2))
    x = GridProperty(ncol=3, nrow=2, nlay=2, values=vals)

    # shall be a maskedarray although input is a np array:
    assert isinstance(x.values, npma.core.MaskedArray)
    assert x.values.mean() == 5.5

    x.values = npma.masked_greater(x.values, 5)
    assert x.values.mean() == 2.5

    # this shall now broadcast the value 33 to all activecells
    x.isdiscrete = True
    x.values = 33
    assert x.dtype == np.int32

    x.isdiscrete = False
    x.values = 44.0221
    assert x.dtype == np.float64


def test_create_actnum():
    """Test creating ACTNUM"""
    x = GridProperty()
    act = x.get_actnum()

    print(x.values)
    print(act.values)
    print(x.nactive)
    print(x.ntotal)

    assert x.nactive < x.ntotal


def test_undef():
    """Test getting UNDEF value"""
    xx = GridProperty()
    act = xx.get_actnum()

    assert xx.undef == xtgeo.UNDEF
    assert act.undef == xtgeo.UNDEF_INT


def test_class_methods():
    """Test getting class methods"""
    result = GridProperty.methods()
    assert "from_file" in result


def test_describe():
    """Test getting the describe text"""
    xx = GridProperty()
    desc = xx.describe(flush=False)
    assert "Name" in desc


def test_npvalues3d():
    """Test getting numpy values as 3d"""
    xx = GridProperty()
    mynp = xx.get_npvalues3d()

    assert mynp.shape == (4, 3, 5)
    assert mynp[0, 0, 0] == xtgeo.UNDEF

    xx.isdiscrete = True
    mynp = xx.get_npvalues3d()
    assert mynp[0, 0, 0] == xtgeo.UNDEF_INT

    mynp2 = xx.get_npvalues3d(fill_value=-999)
    assert mynp2[0, 0, 0] == -999


def test_dtype():
    """Test dtype property"""
    xx = GridProperty()
    act = xx.get_actnum()

    if not xx.isdiscrete:
        xx.dtype = np.float16

    assert xx.dtype == np.float16
    with pytest.raises(ValueError):
        xx.dtype = np.int32

    assert act.dtype == np.int32
    with pytest.raises(ValueError):
        act.dtype = np.float64


def test_create_from_grid():
    """Create a simple property from grid"""

    gg = Grid(TESTFILE5, fformat="egrid")
    poro = GridProperty(gg, name="poro", values=0.33)
    assert poro.ncol == gg.ncol

    assert poro.isdiscrete is False
    assert poro.values.mean() == 0.33

    assert poro.values.dtype.kind == "f"

    faci = xtgeo.GridProperty(gg, name="FAC", values=1, discrete=True)
    assert faci.nlay == gg.nlay
    assert faci.values.mean() == 1

    assert faci.values.dtype.kind == "i"

    some = xtgeo.GridProperty(gg, name="SOME")
    assert some.isdiscrete is False
    some.values = np.where(some.values == 0, 0, 1)
    assert some.isdiscrete is False


def test_create_from_gridproperty():
    """Create a simple property from grid"""

    gg = Grid(TESTFILE5, fformat="egrid")
    poro = GridProperty(gg, name="poro", values=0.33)
    assert poro.ncol == gg.ncol

    # create from gridproperty
    faci = xtgeo.GridProperty(poro, name="FAC", values=1, discrete=True)
    assert faci.nlay == gg.nlay
    assert faci.values.mean() == 1

    assert faci.values.dtype.kind == "i"

    some = xtgeo.GridProperty(faci, name="SOME", values=22)
    assert some.values.mean() == 22.0
    assert some.isdiscrete is False
    some.values = np.where(some.values == 0, 0, 1)
    assert some.isdiscrete is False


def test_pathlib(tmp_path):
    """Import and export via pathlib"""

    pfile = pathlib.Path(DUALROFF)
    grdp = GridProperty()
    grdp.from_file(pfile, name="POROM")

    assert grdp.dimensions == (5, 3, 1)

    out = tmp_path / "grdpathtest.roff"
    grdp.to_file(out, fformat="roff")

    with pytest.raises(OSError):
        out = pathlib.Path() / "nosuchdir" / "grdpathtest.roff"
        grdp.to_file(out, fformat="roff")


def test_roffbin_import1():
    """Test of import of ROFF binary"""

    logger.info("Name is {}".format(__name__))

    x = GridProperty()
    logger.info("Import roff...")
    x.from_file(TESTFILE1, fformat="roff", name="PORO")

    logger.info(repr(x.values))
    logger.info(x.values.dtype)
    logger.info("Porosity is {}".format(x.values))
    logger.info("Mean porosity is {}".format(x.values.mean()))
    assert x.values.mean() == pytest.approx(0.1677, abs=0.001)


def test_roffbin_import1_new():
    """Test ROFF import, new code May 2018"""
    logger.info("Name is {}".format(__name__))

    x = GridProperty()
    logger.info("Import roff...")
    x.from_file(TESTFILE1, fformat="roff", name="PORO")
    logger.info("Porosity is {}".format(x.values))
    logger.info("Mean porosity is {}".format(x.values.mean()))
    assert x.values.mean() == pytest.approx(0.1677, abs=0.001)


def test_roffbin_import2():
    """Import roffbin, with several props in one file."""

    logger.info("Name is {}".format(__name__))
    dz = GridProperty()
    logger.info("Import roff...")
    dz.from_file(TESTFILE2, fformat="roff", name="Z_increment")

    logger.info(repr(dz.values))
    logger.info(dz.values.dtype)
    logger.info("Mean DZ is {}".format(dz.values.mean()))

    hc = GridProperty()
    logger.info("Import roff...")
    hc.from_file(TESTFILE2, fformat="roff", name="Oil_HCPV")

    logger.info(repr(hc.values))
    logger.info(hc.values.dtype)
    logger.info(hc.values3d.shape)
    _ncol, nrow, _nlay = hc.values3d.shape

    assert nrow == 100, "NROW from shape (Emerald)"

    logger.info("Mean HCPV is {}".format(hc.values.mean()))
    assert hc.values.mean() == pytest.approx(1446.4611912446985, abs=0.0001)


def test_eclinit_simple_importexport(tmpdir):
    """Property import and export with anoother name"""

    # let me guess the format (shall be egrid)
    gg = Grid(TESTFILE13A, fformat="egrid")
    po = GridProperty(TESTFILE13B, name="PORO", grid=gg)

    po.to_file(
        os.path.join(tmpdir, "simple.grdecl"),
        fformat="grdecl",
        name="PORO2",
        fmt="%12.5f",
    )

    p2 = GridProperty(os.path.join(tmpdir, "simple.grdecl"), grid=gg, name="PORO2")
    assert p2.name == "PORO2"


def test_grdecl_import_reek(tmpdir):
    """Property GRDECL import from Eclipse. Reek"""

    rgrid = Grid(TESTFILE12A, fformat="grdecl")

    assert rgrid.dimensions == (40, 64, 14)

    poro = GridProperty(TESTFILE12B, name="PORO", fformat="grdecl", grid=rgrid)

    poro2 = GridProperty(TESTFILE1, name="PORO", fformat="roff", grid=rgrid)

    assert poro.values.mean() == pytest.approx(poro2.values.mean(), abs=0.001)
    assert poro.values.std() == pytest.approx(poro2.values.std(), abs=0.001)

    with pytest.raises(KeywordNotFoundError):
        poro3 = GridProperty(TESTFILE12B, name="XPORO", fformat="grdecl", grid=rgrid)
        logger.debug("Keyword failed as expected for instance %s", poro3)

    # Export to ascii grdecl and import that again...
    exportfile = os.path.join(tmpdir, "reekporo.grdecl")
    poro.to_file(exportfile, fformat="grdecl")
    porox = GridProperty(exportfile, name="PORO", fformat="grdecl", grid=rgrid)
    assert poro.values.mean() == pytest.approx(porox.values.mean(), abs=0.001)

    # Export to binary grdecl and import that again...
    exportfile = os.path.join(tmpdir, "reekporo.bgrdecl")
    poro.to_file(exportfile, fformat="bgrdecl")
    porox = GridProperty(exportfile, name="PORO", fformat="bgrdecl", grid=rgrid)
    assert poro.values.mean() == pytest.approx(porox.values.mean(), abs=0.001)


def test_io_roff_discrete(tmpdir):
    """Import ROFF discrete property; then export to ROFF int."""

    logger.info("Name is {}".format(__name__))
    po = GridProperty()
    po.from_file(TESTFILE8, fformat="roff", name="Zone")

    logger.info("\nCodes ({})\n{}".format(po.ncodes, po.codes))

    # tests:
    assert po.ncodes == 3
    logger.debug(po.codes[3])
    assert po.codes[3] == "Below_Low_reek"

    # export discrete to ROFF ...TODO
    po.to_file(
        os.path.join(tmpdir, "reek_zone_export.roff"), name="Zone", fformat="roff"
    )

    # fix some zero values (will not be fixed properly as grid ACTNUM differs?)
    val = po.values
    val = npma.filled(val, fill_value=3)  # trick
    print(val.min(), val.max())
    po.values = val
    print(po.values.min(), po.values.max())
    po.values[:, :, 13] = 1  # just for fun test
    po.to_file(
        os.path.join(tmpdir, "reek_zonefix_export.roff"), name="ZoneFix", fformat="roff"
    )


def test_io_to_nonexisting_folder():
    """Import a prop and try to save in a nonexisting folder"""

    po = GridProperty()
    mygrid = Grid(TESTFILE5)
    po.from_file(TESTFILE7, fformat="unrst", name="PRESSURE", grid=mygrid, date="first")
    with pytest.raises(OSError):
        po.to_file(os.path.join("TMP_NOT", "dummy.grdecl"), fformat="grdecl")


def test_get_all_corners():
    """Get X Y Z for all corners as XTGeo GridProperty objects"""

    grid = Grid(TESTFILE8A)
    allc = grid.get_xyz_corners()

    x0 = allc[0]
    y0 = allc[1]
    z0 = allc[2]
    x1 = allc[3]
    y1 = allc[4]
    z1 = allc[5]

    # top of cell layer 2 in cell 5 5 (if 1 index start as RMS)
    assert x0.values3d[4, 4, 1] == pytest.approx(457387.718, abs=0.5)
    assert y0.values3d[4, 4, 1] == pytest.approx(5935461.29790, abs=0.5)
    assert z0.values3d[4, 4, 1] == pytest.approx(1728.9429, abs=0.1)

    assert x1.values3d[4, 4, 1] == pytest.approx(457526.55367, abs=0.5)
    assert y1.values3d[4, 4, 1] == pytest.approx(5935542.02467, abs=0.5)
    assert z1.values3d[4, 4, 1] == pytest.approx(1728.57898, abs=0.1)


def test_get_cell_corners():
    """Get X Y Z for one cell as tuple"""

    grid = Grid(TESTFILE8A)
    clist = grid.get_xyz_cell_corners(ijk=(4, 4, 1))
    logger.debug(clist)

    assert clist[0] == pytest.approx(457168.358886, abs=0.1)


def test_get_xy_values_for_webportal():
    """Get lists on webportal format"""

    grid = Grid(TESTFILE8A)
    prop = GridProperty(TESTFILE9, grid=grid, name="PORO")

    start = xtg.timer()
    coord, valuelist = prop.get_xy_value_lists(grid=grid)
    elapsed = xtg.timer(start)
    logger.info("Elapsed {}".format(elapsed))
    logger.info("Coords {}".format(coord))

    grid = Grid(TESTFILE10)
    prop = GridProperty(TESTFILE11, grid=grid, name="PORO")

    coord, valuelist = prop.get_xy_value_lists(grid=grid, activeonly=False)

    logger.info("Cell 1 1 1 coords\n{}.".format(coord[0][0]))
    assert coord[0][0][0] == (454.875, 318.5)
    assert valuelist[0][0] == -999.0


def test_get_values_by_ijk():
    """Test getting values for given input arrays for I J K"""
    logger.info("Name is {}".format(__name__))

    x = GridProperty()
    logger.info("Import roff...")
    x.from_file(TESTFILE1, fformat="roff", name="PORO")

    iset1 = np.array([np.nan, 23, 22])
    jset1 = np.array([np.nan, 23, 19])
    kset1 = np.array([np.nan, 13, 2])

    res1 = x.get_values_by_ijk(iset1, jset1, kset1)

    assert res1[1] == pytest.approx(0.08403542, abs=0.0001)
    assert np.isnan(res1[0])


def test_values_in_polygon():
    """Test replace values in polygons"""

    xprop = GridProperty()
    logger.info("Import roff...")
    grid = Grid(TESTFILE5)
    xprop.from_file(TESTFILE1, fformat="roff", name="PORO", grid=grid)
    poly = Polygons(POLYFILE)
    xprop.geometry = grid
    xorig = xprop.copy()

    xprop.operation_polygons(poly, 99, inside=True)
    assert xprop.values.mean() == pytest.approx(25.1788, abs=0.01)

    xp2 = xorig.copy()
    xp2.values *= 100
    xp2.continuous_to_discrete()
    xp2.set_inside(poly, 44)

    xp2.dtype = np.uint8
    xp2.set_inside(poly, 44)
    print(xp2.values)

    xp2.dtype = np.uint16
    xp2.set_inside(poly, 44)
    print(xp2.values)

    xp3 = xorig.copy()
    xp3.values *= 100
    print(xp3.values.mean())
    xp3.dtype = np.float32
    xp3.set_inside(poly, 44)
    print(xp3.values.mean())

    assert xp3.values.mean() == pytest.approx(23.40642788381048, abs=0.001)


@given(dimensions, st.booleans())
@example((4, 3, 5), True)
def test_gridprop_non_default_size_init(dim, discrete):
    ncol, nrow, nlay = dim
    prop = GridProperty(
        ncol=ncol,
        nrow=nrow,
        nlay=nlay,
        discrete=discrete,
    )

    if discrete:
        assert prop.dtype == np.int32
        assert prop.values.dtype == np.int32
    else:
        assert prop.dtype == np.float64
        assert prop.values.dtype == np.float64
    np.testing.assert_allclose(prop.values, np.zeros(dim))


@given(xtgeo_grids, st.booleans())
@example(Grid(), True)
def test_gridprop_grid_init(grid, discrete):
    prop = GridProperty(
        grid,
        discrete=discrete,
    )
    if discrete:
        assert prop.dtype == np.int32
        assert prop.values.dtype == np.int32
    else:
        assert prop.dtype == np.float64
        assert prop.values.dtype == np.float64
    np.testing.assert_allclose(prop.values, np.zeros(grid.dimensions))


@given(dimensions, xtgeo_grids, st.booleans())
@example((4, 3, 5), Grid(), True)
def test_gridprop_grid_and_dim_init(dim, grid, discrete):
    ncol, nrow, nlay = dim
    if dim != grid.dimensions:
        with pytest.raises(ValueError, match="dimension"):
            GridProperty(
                grid,
                ncol=ncol,
                nrow=nrow,
                nlay=nlay,
                discrete=discrete,
            )
    else:
        prop = GridProperty(
            grid,
            ncol=ncol,
            nrow=nrow,
            nlay=nlay,
            discrete=discrete,
        )
        if discrete:
            assert prop.dtype == np.int32
            assert prop.values.dtype == np.int32
        else:
            assert prop.dtype == np.float64
            assert prop.values.dtype == np.float64
        np.testing.assert_allclose(prop.values, np.zeros(dim))


@pytest.mark.parametrize("discrete", [True, False])
def test_gridprop_default(discrete):
    prop = GridProperty(discrete=discrete)

    if discrete:
        assert prop.dtype == np.int32
        assert prop.values.dtype == np.int32
    else:
        assert prop.dtype == np.float64
        assert prop.values.dtype == np.float64

    default_values = np.ma.MaskedArray(np.full((4, 3, 5), 99), False)
    default_values[0:4, 0, 0:2] = np.ma.masked

    np.testing.assert_allclose(prop.values, default_values)


@pytest.mark.parametrize("discrete", [True, False])
def test_gridprop_values_and_discrete_init(discrete):
    prop = GridProperty(discrete=discrete, values=np.zeros((4, 3, 5)))

    if discrete:
        assert prop.dtype == np.int32
        assert prop.values.dtype == np.int32
    else:
        assert prop.dtype == np.float64
        assert prop.values.dtype == np.float64

    np.testing.assert_allclose(prop.values, np.ma.zeros((4, 3, 5)))


def test_gridprop_init_default_with_value():
    prop = GridProperty(discrete=True, values=1)
    assert np.array_equal(prop.values, np.ma.ones(shape=(4, 3, 5), dtype=np.int32))


@given(
    st.sampled_from([np.uint16, np.uint8, np.float32]),
    st.booleans(),
    st.one_of(st.integers(), st.floats()),
)
def test_gridprop_no_override_roxar_dtype(roxar_dtype, discrete, val):
    prop = GridProperty(
        values=val,
        discrete=discrete,
        roxar_dtype=roxar_dtype,
    )
    assert prop.roxar_dtype == roxar_dtype


def test_gridprop_init_roxar_dtype():
    assert GridProperty(discrete=True).roxar_dtype == np.uint8
    assert GridProperty(discrete=False).roxar_dtype == np.float32
    with pytest.raises(ValueError, match="roxar_dtype"):
        GridProperty(roxar_dtype=np.int64)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("fformat", ["grdecl", "bgrdecl"])
@given(xtgeo_grids)
def test_gridprop_export_actnum(fformat, tmp_path, grid):
    actnum = grid.get_actnum(asmasked=True)

    filepath = tmp_path / "actnum.DATA"
    actnum.to_file(filepath, fformat=fformat)

    actnum2 = xtgeo.gridproperty_from_file(
        filepath, name="ACTNUM", fformat=fformat, grid=grid
    )
    if fformat == "grdecl":
        actnum2.isdiscrete = True

    assert actnum.values.tolist() == actnum2.values.tolist()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("fformat", ["grdecl", "bgrdecl"])
@given(xtgeo_grids)
def test_gridprop_export_actnum_append(fformat, tmp_path, grid):
    actnum = grid.get_actnum(asmasked=True)

    filepath = tmp_path / "actnum.DATA"
    actnum.to_file(filepath, name="ACTNUM2", fformat=fformat)
    actnum.to_file(filepath, name="ACTNUM3", append=True, fformat=fformat)

    actnum2 = xtgeo.gridproperty_from_file(
        filepath, name="ACTNUM2", fformat=fformat, grid=grid
    )
    actnum3 = xtgeo.gridproperty_from_file(
        filepath, name="ACTNUM3", fformat=fformat, grid=grid
    )

    if fformat == "grdecl":
        actnum2.isdiscrete = True
        actnum3.isdiscrete = True

    assert actnum.values.tolist() == actnum2.values.tolist()
    assert actnum2.values.tolist() == actnum3.values.tolist()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(xtgeo_grids)
def test_gridprop_export_bgrdecl_double(tmp_path, grid):
    actnum = grid.get_actnum(asmasked=True)

    actnum.to_file(tmp_path / "actnum.DATA", fformat="bgrdecl", dtype=np.float64)

    with open(tmp_path / "actnum.DATA", "rb") as fh:
        assert b"DOUB" in fh.read()
