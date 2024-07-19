"""Testing: test_grid_operations"""

import io
import pathlib

import hypothesis.strategies as st
import pytest
import xtgeo
from hypothesis import assume, given
from xtgeo.common import XTGeoDialog
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.grid3d import GridProperties, GridProperty

from .grid_generator import xtgeo_grids
from .gridprop_generator import grid_properties as gridproperties_elements, keywords

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

GFILE1 = pathlib.Path("3dgrids/reek/REEK.EGRID")
IFILE1 = pathlib.Path("3dgrids/reek/REEK.INIT")
RFILE1 = pathlib.Path("3dgrids/reek/REEK.UNRST")
RFILE2 = pathlib.Path(
    "3dgrids/simpleb8/E100_3LETTER_TRACER.UNRST"
)  # has kword with spaces
SPEFILE1 = pathlib.Path("3dgrids/bench_spe9/BENCH_SPE9.UNRST")

XFILE2 = pathlib.Path("3dgrids/reek/reek_grd_w_props.roff")


@st.composite
def gridproperties(draw):
    grid = draw(xtgeo_grids)
    names = draw(st.lists(keywords, unique=True))
    return GridProperties(
        props=[
            draw(gridproperties_elements(name=st.just(n), grid=st.just(grid)))
            for n in names
        ]
    )


@given(gridproperties(), st.text())
def test_gridproperties_get_prop_by_name_not_exists(gps, name):
    assume(name not in gps.names)

    assert gps.get_prop_by_name(name, raiseserror=False) is None

    with pytest.raises(ValueError, match="Cannot find"):
        gps.get_prop_by_name(name)

    with pytest.raises(KeyError, match="does not exist"):
        gps[name]


def test_gridproperties_import_date_does_not_exist(tmp_path):
    tmpfile = tmp_path / "TEST.UNRST"
    tmpfile.write_text("")
    with pytest.raises(ValueError, match="dates are either"):
        xtgeo.gridproperties_from_file(
            tmpfile,
            names="all",
            dates=["12.12.1"],
            fformat="unrst",
            grid=xtgeo.create_box_grid((2, 2, 2)),
        )


def test_gridproperties_iter(testdata_path):
    g = xtgeo.grid_from_file(testdata_path / GFILE1, fformat="egrid")

    gps = xtgeo.gridproperties_from_file(
        testdata_path / IFILE1, fformat="init", names=["PORO", "PORV"], grid=g
    )

    count = 0
    for _ in gps:
        for _ in gps:
            count += 1

    assert count == 4


@given(gridproperties())
def test_gridproperties_copy(grid_properties):
    grid_properties_copy = grid_properties.copy()

    assert grid_properties_copy.dates == grid_properties.dates
    assert grid_properties_copy.names == grid_properties.names

    assume(grid_properties.props)
    assume(grid_properties.props[0].date != 19990101)
    grid_properties_copy.props[0].date = 19990101
    assert grid_properties.props[0].date != 19990101


@given(gridproperties_elements(), gridproperties_elements())
def test_consistency_check(gridproperty1, gridproperty2):
    assume(gridproperty1.dimensions != gridproperty2.dimensions)
    with pytest.raises(ValueError, match="Mismatching dimensions"):
        GridProperties(props=[gridproperty1, gridproperty2])


@given(gridproperties_elements())
def test_gridproperties_from_roff(grid_property):
    buff = io.BytesIO()
    grid_property.to_file(buff, fformat="roff")
    buff.seek(0)
    props = xtgeo.gridproperties_from_file(
        buff, fformat="roff", names=[grid_property.name]
    )

    assert props.names == [grid_property.name]


@given(gridproperties_elements())
def test_gridproperties_invalid_format(grid_property):
    buff = io.BytesIO()
    grid_property.to_file(buff, fformat="roff")
    with pytest.raises(InvalidFileFormatError, match="invalid for type GridProperties"):
        xtgeo.gridproperties_from_file(buff, fformat="segy")


def test_scan_dates(testdata_path):
    """A static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    assert GridProperties.scan_dates(testdata_path / RFILE2) == [(0, 20220101)]
    t2 = xtg.timer(t1)
    logger.info(f"Scanned {RFILE2} scanned in {t2} seconds")

    t1 = xtg.timer()
    assert GridProperties.scan_dates(testdata_path / RFILE1) == [
        (0, 19991201),
        (1, 20000101),
        (2, 20000201),
        (3, 20000301),
        (4, 20000401),
        (5, 20000501),
        (6, 20000601),
        (7, 20000701),
        (8, 20000801),
        (9, 20000901),
        (10, 20001001),
        (11, 20001101),
        (12, 20001201),
        (13, 20010101),
        (14, 20030101),
    ]
    t2 = xtg.timer(t1)
    logger.info(f"Scanned {RFILE1} scanned in {t2} seconds")
    t1 = xtg.timer()
    assert GridProperties.scan_dates(testdata_path / SPEFILE1) == [
        (0, 19900101),
        (1, 19900102),
        (2, 19900103),
        (3, 19900105),
        (4, 19900109),
        (5, 19900117),
        (6, 19900129),
        (7, 19900210),
        (8, 19900302),
        (9, 19900322),
        (10, 19900411),
        (11, 19900426),
        (12, 19900521),
        (13, 19900620),
        (14, 19900720),
        (15, 19900908),
        (16, 19901028),
        (17, 19901117),
        (18, 19901207),
        (19, 19901227),
        (20, 19910101),
        (21, 19910111),
        (22, 19910121),
        (23, 19910205),
        (24, 19910225),
        (25, 19910327),
        (26, 19910426),
        (27, 19910526),
        (28, 19910625),
        (29, 19910824),
        (30, 19911023),
        (31, 19911222),
        (32, 19920205),
        (33, 19920321),
        (34, 19920505),
        (35, 19920619),
    ]
    t2 = xtg.timer(t1)
    logger.info(f"Scanned {SPEFILE1} scanned in {t2} seconds")


def test_scan_dates_invalid_file(testdata_path):
    """Raise an error before trying to scan a non-existent file."""
    with pytest.raises(ValueError, match="does not exist"):
        GridProperties.scan_dates(testdata_path / pathlib.Path("notafile.UNRST"))


def test_dates_from_restart(testdata_path):
    """A simpler static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    assert GridProperties.scan_dates(testdata_path / RFILE2, datesonly=True) == [
        20220101
    ]
    t2 = xtg.timer(t1)
    logger.info(f"Scanned {RFILE2} scanned in {t2} seconds")
    t1 = xtg.timer()
    assert GridProperties.scan_dates(testdata_path / RFILE1, datesonly=True) == [
        19991201,
        20000101,
        20000201,
        20000301,
        20000401,
        20000501,
        20000601,
        20000701,
        20000801,
        20000901,
        20001001,
        20001101,
        20001201,
        20010101,
        20030101,
    ]
    t2 = xtg.timer(t1)
    logger.info(f"Scanned {RFILE1} scanned in {t2} seconds")
    t1 = xtg.timer()
    assert GridProperties.scan_dates(testdata_path / SPEFILE1, datesonly=True) == [
        19900101,
        19900102,
        19900103,
        19900105,
        19900109,
        19900117,
        19900129,
        19900210,
        19900302,
        19900322,
        19900411,
        19900426,
        19900521,
        19900620,
        19900720,
        19900908,
        19901028,
        19901117,
        19901207,
        19901227,
        19910101,
        19910111,
        19910121,
        19910205,
        19910225,
        19910327,
        19910426,
        19910526,
        19910625,
        19910824,
        19911023,
        19911222,
        19920205,
        19920321,
        19920505,
        19920619,
    ]
    t2 = xtg.timer(t1)
    logger.info(f"Scanned {SPEFILE1} scanned in {t2} seconds")


def test_get_dataframe(testdata_path):
    """Get a Pandas dataframe from the gridproperties"""

    g = xtgeo.grid_from_file(testdata_path / GFILE1, fformat="egrid")

    names = ["SOIL", "SWAT", "PRESSURE"]
    dates = [19991201]
    x = xtgeo.gridproperties_from_file(
        testdata_path / RFILE1, fformat="unrst", names=names, dates=dates, grid=g
    )
    df = x.get_dataframe(activeonly=True, ijk=True, xyz=False)

    print(df.head())

    assert df["SWAT_19991201"].mean() == pytest.approx(0.87802, abs=0.001)
    assert df["PRESSURE_19991201"].mean() == pytest.approx(334.523, abs=0.005)


def test_get_dataframe_active_only(testdata_path):
    """Get a Pandas dataframe from the gridproperties"""

    grid = xtgeo.grid_from_file(testdata_path / GFILE1, fformat="egrid")
    gps = xtgeo.gridproperties_from_file(
        testdata_path / RFILE1,
        fformat="unrst",
        names=["SOIL", "SWAT", "PRESSURE"],
        dates=[19991201],
        grid=grid,
    )

    df = gps.get_dataframe(activeonly=True, ijk=True, xyz=False)
    assert len(df.index == grid.nactive)

    df2 = xtgeo.gridproperties_dataframe(gps, activeonly=True, ijk=True, xyz=False)
    assert (df == df2).all().all()


def test_gridproperties_all_roff(testdata_path):
    """Read all gridproperties from ROFF binary format."""

    gps = xtgeo.gridproperties_from_file(
        testdata_path / XFILE2,
        fformat="roff",
        names="all",
    )

    for name in ("PORV", "PORO", "EQLNUM", "FIPNUM"):
        assert name in gps.names


def test_gridproperties_read_roff_missing_name(testdata_path):
    """Read gridproperties from ROFF binary format, with one key not present."""

    with pytest.raises(ValueError):
        gps = xtgeo.gridproperties_from_file(
            testdata_path / XFILE2,
            fformat="roff",
            names=["PORO", "EQLNUM", "NOTPRESENT"],
            strict=True,
        )

    gps = xtgeo.gridproperties_from_file(
        testdata_path / XFILE2,
        fformat="roff",
        names=["PORO", "EQLNUM", "NOTPRESENT"],
        strict=False,
    )

    assert "PORO" in gps.names
    assert "NOTPRESENT" not in gps.names


@given(gridproperties())
def test_get_dataframe_no_grid(gridproperties):
    with pytest.raises(ValueError, match="no Grid is present"):
        xtgeo.gridproperties_dataframe(gridproperties, ijk=True, activeonly=False)

    with pytest.raises(ValueError, match="no Grid is present"):
        xtgeo.gridproperties_dataframe(
            gridproperties, ijk=True, activeonly=True, xyz=True
        )


@given(gridproperties())
def test_get_dataframe_filled(gridproperties):
    gridproperties_list = list(gridproperties)
    assume(len(gridproperties_list) > 0)
    df = xtgeo.gridproperties_dataframe(gridproperties, ijk=False, activeonly=False)
    assert (
        len(df.index) == gridproperties.ncol * gridproperties.nrow * gridproperties.nlay
    )


def test_props_set_get() -> None:
    gp = GridProperties()
    assert gp.props is None

    props = [GridProperty()]
    gp = GridProperties()
    gp.props = props
    assert gp.props == props

    props = [GridProperty()]
    gp = GridProperties(props=props)
    assert gp.props == props
