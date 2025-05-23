import logging
import pathlib
from pathlib import Path

import pytest

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.grid3d import list_gridproperties
from xtgeo.grid3d._gridprops_import_roff import read_roff_properties
from xtgeo.io._file import FileWrapper

logger = logging.getLogger(__name__)

E100_BO_FINIT = pathlib.Path("3dgrids/simpleb8/E100_BO.FINIT")
E100_BO_FUNRST = pathlib.Path("3dgrids/simpleb8/E100_BO.FUNRST")
SPE_UNRST = pathlib.Path("3dgrids/bench_spe9/BENCH_SPE9.UNRST")
SPE_INIT = pathlib.Path("3dgrids/bench_spe9/BENCH_SPE9.INIT")
REEK_INIT = pathlib.Path("3dgrids/reek/REEK.INIT")
REEK_UNRST = pathlib.Path("3dgrids/reek/REEK.UNRST")

REEK_SIM_PORO = pathlib.Path("3dgrids/reek/reek_sim_poro.roff")
ROFF_PROPS = pathlib.Path("3dgrids/reek/reek_grd_w_props.roff")
ROFFASC_PROPS = pathlib.Path("3dgrids/reek/reek_grd_w_props.roffasc")
ROFF_THREE_PROPS = pathlib.Path("3dgrids/reek/reek_geo2_grid_3props.roff")


@pytest.mark.parametrize("test_file", ["A.EGRID", "b.grdecl", "t.segy", "c.rms_attr"])
def test_raise_on_invalid_filetype(tmp_path, test_file):
    filepath = tmp_path / test_file
    Path(filepath).touch()
    with pytest.raises(InvalidFileFormatError, match="invalid for type GridProperties"):
        list_gridproperties(filepath)


@pytest.mark.parametrize(
    "test_file, expected",
    [
        (REEK_SIM_PORO, ["PORO"]),
        (ROFF_PROPS, ["PORV", "PORO", "EQLNUM", "FIPNUM"]),
        (ROFF_THREE_PROPS, ["Poro", "EQLNUM", "Facies"]),
    ],
)
def test_read_roff_properties(testdata_path, test_file, expected):
    xtg_file = FileWrapper(testdata_path / test_file)
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
def test_list_properties_from_roff(testdata_path, test_file, expected):
    assert list_gridproperties(testdata_path / test_file) == expected


@pytest.mark.parametrize("test_file", [SPE_INIT, REEK_INIT, E100_BO_FINIT])
def test_list_properties_from_init(testdata_path, test_file):
    props = list_gridproperties(testdata_path / test_file)
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
def test_list_properties_from_unrst(testdata_path, test_file):
    props = list_gridproperties(testdata_path / test_file)
    # Just some common dynamic properties
    for prop in ("PRESSURE", "SWAT", "SGAS", "RS"):
        assert prop in props
