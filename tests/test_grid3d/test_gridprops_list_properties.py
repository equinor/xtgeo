import sys
from pathlib import Path

import pytest
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import list_gridproperties
from xtgeo.grid3d._gridprops_import_roff import read_roff_properties
from xtgeo.io._file_wrapper import FileWrapper

xtg = XTGeoDialog()

if not xtg.testsetup():
    sys.exit(-9)

TPATH = xtg.testpathobj

logger = xtg.basiclogger(__name__)

E100_BO_FINIT = TPATH / "3dgrids/simpleb8/E100_BO.FINIT"
E100_BO_FUNRST = TPATH / "3dgrids/simpleb8/E100_BO.FUNRST"
SPE_UNRST = TPATH / "3dgrids/bench_spe9/BENCH_SPE9.UNRST"
SPE_INIT = TPATH / "3dgrids/bench_spe9/BENCH_SPE9.INIT"
REEK_INIT = TPATH / "3dgrids/reek/REEK.INIT"
REEK_UNRST = TPATH / "3dgrids/reek/REEK.UNRST"

REEK_SIM_PORO = TPATH / "3dgrids/reek/reek_sim_poro.roff"
ROFF_PROPS = TPATH / "3dgrids/reek/reek_grd_w_props.roff"
ROFFASC_PROPS = TPATH / "3dgrids/reek/reek_grd_w_props.roffasc"
ROFF_THREE_PROPS = TPATH / "3dgrids/reek/reek_geo2_grid_3props.roff"


@pytest.mark.parametrize("test_file", ["A.EGRID", "b.grdecl", "t.segy", "c.RSSPEC"])
def test_raise_on_invalid_filetype(tmp_path, test_file):
    filepath = tmp_path / test_file
    Path(filepath).touch()
    with pytest.raises(ValueError, match="file format"):
        list_gridproperties(filepath)


@pytest.mark.parametrize(
    "test_file, expected",
    [
        (REEK_SIM_PORO, ["PORO"]),
        (ROFF_PROPS, ["PORV", "PORO", "EQLNUM", "FIPNUM"]),
        (ROFF_THREE_PROPS, ["Poro", "EQLNUM", "Facies"]),
    ],
)
def test_read_roff_properties(test_file, expected):
    xtg_file = FileWrapper(test_file)
    assert list(read_roff_properties(xtg_file)) == expected


@pytest.mark.parametrize(
    "test_file, expected",
    [
        (REEK_SIM_PORO, ["PORO"]),
        (ROFF_PROPS, ["PORV", "PORO", "EQLNUM", "FIPNUM"]),
        # ROFFASC_PROPS is slow
        (ROFFASC_PROPS, ["PORV", "PORO", "EQLNUM", "FIPNUM"]),
        (ROFF_THREE_PROPS, ["Poro", "EQLNUM", "Facies"]),
    ],
)
def test_list_properties_from_roff(test_file, expected):
    assert list_gridproperties(test_file) == expected


@pytest.mark.parametrize("test_file", [SPE_INIT, REEK_INIT, E100_BO_FINIT])
def test_list_properties_from_init(test_file):
    props = list_gridproperties(test_file)
    # Just some common static properties
    for prop in (
        "PORV",
        "DX",
        "DY",
        "DZ",
        "PERMX",
        "PERMY",
        "PERMZ",
        "EQLNUM",
        "FIPNUM",
    ):
        assert prop in props


@pytest.mark.parametrize("test_file", [SPE_UNRST, REEK_UNRST, E100_BO_FUNRST])
def test_list_properties_from_unrst(test_file):
    props = list_gridproperties(test_file)
    # Just some common dynamic properties
    for prop in ("PRESSURE", "SWAT", "SGAS", "RS"):
        assert prop in props
