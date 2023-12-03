"""Test basic resfo wrt to xtgeo testdata and core resfo and xtgeo usage.

Based on 'simpleb8' and ran through various simulators in 2022 by V. Kippe (Equinor):

    E100_BO: Eclipse 100 blackoil (oil/wat/gas + disgas/vapoil)

    E300_BO: Eclipse 300 blackoil

    IX_BO: IX blackoil, EGRID genererated by ‘migrator’ (IX = InterSect)

    IX_BO_GRIDREPORT: IX blackoil, with EGRID from IX itself (may differ!)

    E300_COMP: Eclipse 300 compositional

    IX_COMP: IX compositional, with EGRID from migrator

    IX_COMP_GRIDREPORT: IX compositional, with EGRID from IX
"""
import numpy as np
import pytest
import resfo
import xtgeo

xtg = xtgeo.XTGeoDialog()

logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

SIMPLEB8_PATH = TPATH / "3dgrids/simpleb8"


@pytest.mark.parametrize(
    "ecl_like_file, some_valid_kwords, some_notpresent_kwords",
    [
        (
            "E100_BO.EGRID",
            ["FILEHEAD", "MAPAXES", "ZCORN", "NNC1"],
            [],
        ),
        (
            "E100_BO.GRID",
            ["DIMENS", "GRIDUNIT", "COORDS"],
            ["FILEHEAD"],
        ),
        (
            "E300_BO.EGRID",
            ["FILEHEAD", "MAPAXES", "ZCORN", "NNC1"],
            [],
        ),
        (
            "E300_BO.GRID",
            ["DIMENS", "GRIDUNIT", "COORDS"],
            ["FILEHEAD"],
        ),
        (
            "E300_COMP.EGRID",
            ["FILEHEAD", "MAPAXES", "ZCORN", "NNC1"],
            [],
        ),
        (
            "IX_BO.EGRID",
            ["FILEHEAD", "MAPAXES", "ZCORN", "NNC1"],
            [],
        ),
        (
            "IX_BO.GRID",
            ["DIMENS", "GRIDUNIT", "COORDS"],
            ["FILEHEAD"],
        ),
        (
            "IX_COMP.EGRID",
            ["FILEHEAD", "MAPAXES", "ZCORN", "NNC1"],
            [],
        ),
        (
            "IX_COMP_GRIDREPORT.EGRID",
            ["FILEHEAD", "MAPAXES", "ZCORN"],
            ["NNC1"],  # NNC1 is not in IX "GRIDREPORT" for some reason
        ),
    ],
)
def test_refo_read_grids(ecl_like_file, some_valid_kwords, some_notpresent_kwords):
    """Read grid data, different simulators."""
    kwords = []
    ktypes = []
    for item in resfo.lazy_read(SIMPLEB8_PATH / ecl_like_file):
        kwords.append(item.read_keyword().strip())
        ktypes.append(item.read_type().strip())

    for kword in some_valid_kwords:
        assert kword in kwords

    for kword in some_notpresent_kwords:
        assert kword not in kwords


@pytest.mark.parametrize(
    "ecl_like_file, some_valid_kwords, some_notpresent_kwords",
    [
        (
            "E100_BO.INIT",
            ["INTEHEAD", "PORO", "PORV"],
            [],
        ),
        (
            "E300_BO.INIT",
            ["INTEHEAD", "PORO", "PORV"],
            [],
        ),
        (
            "IX_BO.INIT",
            ["INTEHEAD", "PORO", "PORV"],
            [],
        ),
    ],
)
def test_refo_read_init(ecl_like_file, some_valid_kwords, some_notpresent_kwords):
    """Read INIT data, different simulators."""
    kwords = []
    ktypes = []
    for item in resfo.lazy_read(SIMPLEB8_PATH / ecl_like_file):
        kwords.append(item.read_keyword().strip())
        ktypes.append(item.read_type().strip())

    for kword in some_valid_kwords:
        assert kword in kwords

    for kword in some_notpresent_kwords:
        assert kword not in kwords


@pytest.mark.parametrize(
    "ecl_like_file, some_valid_kwords, some_notpresent_kwords",
    [
        (
            "E100_BO.UNRST",
            ["INTEHEAD", "PRESSURE", "SWAT", "HIDDEN"],
            ["SOIL"],
        ),
        (
            "E300_BO.UNRST",
            ["INTEHEAD", "PRESSURE", "SWAT", "SOIL", "ZPHASE"],
            ["HIDDEN"],
        ),
        (
            "E300_COMP.UNRST",
            ["INTEHEAD", "PRESSURE", "SWAT", "SOIL", "ZPHASE"],
            ["HIDDEN"],
        ),
        (
            "IX_BO.UNRST",
            ["INTEHEAD", "PRESSURE", "SWAT", "SOIL"],
            ["HIDDEN", "ZPHASE"],
        ),
    ],
)
def test_refo_read_restart(ecl_like_file, some_valid_kwords, some_notpresent_kwords):
    """Read UNRST data, different simulators."""
    kwords = []
    ktypes = []
    for item in resfo.lazy_read(SIMPLEB8_PATH / ecl_like_file):
        kwords.append(item.read_keyword().strip())
        ktypes.append(item.read_type().strip())

    for kword in some_valid_kwords:
        assert kword in kwords

    for kword in some_notpresent_kwords:
        assert kword not in kwords

    print(kwords)


@pytest.mark.parametrize(
    "ecl_like_grid, ecl_like_init, keywords, expected_averages",
    [
        (
            "E100_BO.EGRID",
            "E100_BO.INIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
        (
            "E100_BO.FEGRID",
            "E100_BO.FINIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
        (
            "E300_BO.EGRID",
            "E300_BO.INIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
        (
            "E300_COMP.EGRID",
            "E300_COMP.INIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
        (
            "IX_BO.EGRID",
            "IX_BO.INIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
        (
            "IX_COMP.EGRID",
            "IX_COMP.INIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
        (
            "IX_COMP_GRIDREPORT.EGRID",
            "IX_COMP_GRIDREPORT.INIT",
            ["PORO", "TRANZ"],
            [0.257455, 571.378237],
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_egrid_init_read_xtgeo(
    ecl_like_grid,
    ecl_like_init,
    keywords,
    expected_averages,
):
    """Test reading properties from various simulators, egrid and init.

    Internally, xtgeo uses refo for Eclipse like data
    """
    grid = xtgeo.grid_from_file(SIMPLEB8_PATH / ecl_like_grid)
    actnum = grid.get_actnum_indices()
    new_actnum = grid.get_actnum_indices()
    assert np.array_equal(actnum, new_actnum)
    for num, keyw in enumerate(keywords):
        prop = xtgeo.gridproperty_from_file(
            SIMPLEB8_PATH / ecl_like_init, name=keyw, grid=grid
        )
        assert prop.values.mean() == pytest.approx(expected_averages[num], rel=0.001)


@pytest.mark.parametrize(
    "ecl_like_grid, ecl_like_unrst, keywords, date, expected_averages",
    [
        (
            "E100_BO.EGRID",
            "E100_BO.UNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [270.0358789, 0.98801123],
        ),
        (
            "E100_BO.FEGRID",
            "E100_BO.FUNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [270.0358789, 0.98801123],
        ),
        (
            "E300_BO.EGRID",
            "E300_BO.UNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [270.0358789, 0.98801123],
        ),
        (
            "E300_COMP.EGRID",
            "E300_COMP.UNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [400.72074, 0.951075],
        ),
        (
            "IX_BO.EGRID",
            "IX_BO.UNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [270.0358789, 0.98801123],
        ),
        (
            "IX_COMP.EGRID",
            "IX_COMP.UNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [400.72074, 0.951075],
        ),
        (
            "IX_COMP_GRIDREPORT.EGRID",
            "IX_COMP_GRIDREPORT.UNRST",
            ["PRESSURE", "SWAT"],
            20220101,
            [400.72074, 0.951075],
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_egrid_unrst_read_xtgeo(
    ecl_like_grid,
    ecl_like_unrst,
    keywords,
    date,
    expected_averages,
):
    """Test reading properties from various simulators, egrid and unrst

    Internally, xtgeo uses refo for Eclipse like data
    """
    grid = xtgeo.grid_from_file(SIMPLEB8_PATH / ecl_like_grid)

    for num, keyw in enumerate(keywords):
        prop = xtgeo.gridproperty_from_file(
            SIMPLEB8_PATH / ecl_like_unrst, name=keyw, date=date, grid=grid
        )

        assert prop.values.mean() == pytest.approx(expected_averages[num], rel=0.001)
