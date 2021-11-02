import io

import ecl_data_io as eclio
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

import xtgeo as xtg
import xtgeo.grid3d._egrid as xtge

from .egrid_generator import (
    egrids,
    lgr_sections,
    xtgeo_compatible_egridheads,
    xtgeo_compatible_egrids,
)
from .grid_generator import xtgeo_grids


@pytest.mark.parametrize(
    "file_contents, bad_keyword",
    [
        ({"FILEHEAD": []}, "FILEHEAD"),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "MAPUNITS": [],
            },
            "MAPUNITS",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "MAPAXES ": [],
            },
            "MAPAXES",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": [],
            },
            "GRIDHEAD",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
            },
            "COORD",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
            },
            "ZCORN",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
                "ZCORN   ": [],
                "ACTNUM  ": [],
            },
            "ENDGRID",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
                "ZCORN   ": [],
                "ACTNUM  ": [],
                "ENDGRID ": [],
                "NNCHEAD ": [],
            },
            "NNCHEAD",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
                "ZCORN   ": [],
                "ACTNUM  ": [],
                "ENDGRID ": [],
                "NNCHEAD ": np.array([1, 0], dtype=np.int32),
            },
            "NNC1",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
                "ZCORN   ": [],
                "ACTNUM  ": [],
                "ENDGRID ": [],
                "NNCHEAD ": np.array([1, 0], dtype=np.int32),
                "NNC1    ": [],
            },
            "NNC2",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
                "ZCORN   ": [],
                "ACTNUM  ": [],
                "ENDGRID ": [],
                "LGR     ": [],
            },
            "LGR",
        ),
        (
            {
                "FILEHEAD": np.zeros((100,), dtype=np.int32),
                "GRIDUNIT": ["METRES  "],
                "GRIDHEAD": np.ones((100,), dtype=np.int32),
                "COORD   ": [],
                "ZCORN   ": [],
                "ACTNUM  ": [],
                "ENDGRID ": [],
                "LGR     ": ["name"],
            },
            "GRIDHEAD",
        ),
        (
            [
                ("FILEHEAD", np.zeros((100,), dtype=np.int32)),
                ("GRIDUNIT", ["METRES  "]),
                ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
                ("COORD   ", []),
                ("ZCORN   ", []),
                ("ACTNUM  ", []),
                ("ENDGRID ", []),
                ("LGR     ", ["name"]),
                ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
            ],
            "COORD",
        ),
        (
            [
                ("FILEHEAD", np.zeros((100,), dtype=np.int32)),
                ("GRIDUNIT", ["METRES  "]),
                ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
                ("COORD   ", []),
                ("ZCORN   ", []),
                ("ACTNUM  ", []),
                ("ENDGRID ", []),
                ("LGR     ", ["name"]),
                ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
                ("COORD   ", []),
            ],
            "ZCORN",
        ),
        (
            [
                ("FILEHEAD", np.zeros((100,), dtype=np.int32)),
                ("GRIDUNIT", ["METRES  "]),
                ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
                ("COORD   ", []),
                ("ZCORN   ", []),
                ("ACTNUM  ", []),
                ("ENDGRID ", []),
                ("LGR     ", ["name"]),
                ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
                ("COORD   ", []),
                ("ZCORN   ", []),
                ("HOSTNUM ", []),
            ],
            "ENDLGR",
        ),
    ],
)
def test_bad_keywords_raises(file_contents, bad_keyword):
    buf = io.BytesIO()
    eclio.write(buf, file_contents)
    buf.seek(0)
    with pytest.raises(xtge.EGridFileFormatError, match=bad_keyword):
        xtge.EGrid.from_file(buf)


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(egrids())
def test_egrid_read_write(tmp_path, egrid):
    tmp_file = tmp_path / "grid.EGRID"
    egrid.to_file(tmp_file)
    assert xtge.EGrid.from_file(tmp_file) == egrid


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(xtgeo_compatible_egrids(head=xtgeo_compatible_egridheads(mpaxes=st.just(None))))
def test_egrid_from_xtgeo(tmp_path, egrid):
    tmp_file = tmp_path / "grid.EGRID"
    egrid.to_file(tmp_file)
    xtgeo_grid = xtg.grid_from_file(tmp_file, relative_to=xtge.GridRelative.ORIGIN)
    roundtrip_grid = xtge.EGrid.from_xtgeo_grid(xtgeo_grid)
    assert roundtrip_grid.zcorn.tolist() == egrid.zcorn.tolist()
    assert roundtrip_grid.coord.tolist() == egrid.coord.tolist()
    if roundtrip_grid.actnum is None:
        assert egrid.actnum is None or all(egrid.actnum == 1)
    elif egrid.actnum is None:
        assert roundtrip_grid.actnum is None or all(roundtrip_grid.actnum == 1)
    else:
        assert roundtrip_grid.actnum.tolist() == egrid.actnum.tolist()


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(xtgeo_grids)
def test_egrid_to_xtgeo(tmp_path, xtg_grid):
    tmp_file = tmp_path / "grid.EGRID"
    xtg_grid.to_file(tmp_file, fformat="egrid")
    roundtrip_grid = xtge.EGrid.from_file(tmp_file)
    original_grid = xtge.EGrid.from_xtgeo_grid(xtg_grid)
    assert roundtrip_grid.zcorn.tolist() == pytest.approx(original_grid.zcorn.tolist())
    assert roundtrip_grid.coord.tolist() == pytest.approx(original_grid.coord.tolist())
    if roundtrip_grid.actnum is None:
        assert original_grid.actnum is None or all(original_grid.actnum == 1)
    elif original_grid.actnum is None:
        assert roundtrip_grid.actnum is None or all(roundtrip_grid.actnum == 1)
    else:
        assert roundtrip_grid.actnum.tolist() == original_grid.actnum.tolist()


@pytest.mark.parametrize("egrid_type_value", [0, 1, 2])
def test_to_from_filehead_type(egrid_type_value):
    values = np.zeros((100,), dtype=np.int32)
    values[4] = egrid_type_value
    assert xtge.Filehead.from_egrid(values).to_egrid()[4] == egrid_type_value


@given(
    st.sampled_from(xtge.TypeOfGrid),
    st.sampled_from(xtge.RockModel),
    st.sampled_from(xtge.GridFormat),
)
def test_from_to_filehead_type(type_of_grid, rock_model, grid_format):
    filehead = xtge.Filehead(3, 2007, 2, type_of_grid, rock_model, grid_format)
    filehead_roundtrip = xtge.Filehead.from_egrid(filehead.to_egrid())

    assert filehead_roundtrip.year == 2007
    assert filehead_roundtrip.version_number == 3
    assert filehead_roundtrip.version_bound == 2
    assert filehead_roundtrip.type_of_grid == type_of_grid
    assert filehead_roundtrip.rock_model == rock_model
    assert filehead_roundtrip.grid_format == grid_format


def test_type_of_grid_error():
    with pytest.raises(ValueError, match="grid type"):
        xtge.Filehead.from_egrid([3, 2007, 0, 2, 4, 0, 0])


def test_file_head_error():
    with pytest.raises(ValueError, match="too few values"):
        xtge.Filehead.from_egrid([])


def test_grid_head_error():
    with pytest.raises(ValueError, match="Too few arguments"):
        xtge.GridHead.from_egrid([])


def test_read_duplicate_keyword_error():
    buf = io.BytesIO()
    eclio.write(buf, [("FILEHEAD", np.zeros((100,), dtype=np.int32))] * 2)
    buf.seek(0)
    reader = xtge.EGridReader(buf)

    with pytest.raises(xtge.EGridFileFormatError, match="Duplicate"):
        reader.read()


def test_read_bad_keyword_error():
    buf = io.BytesIO()
    eclio.write(buf, [("NTKEYWRD", np.zeros((100,), dtype=np.int32))] * 2)
    buf.seek(0)
    reader = xtge.EGridReader(buf)

    with pytest.raises(xtge.EGridFileFormatError, match="Unknown egrid keyword"):
        reader.read()


def test_read_mixed_gridhead():
    buf = io.BytesIO()
    eclio.write(
        buf,
        [
            ("FILEHEAD", np.zeros((100,), dtype=np.int32)),
            ("GRIDUNIT", ["METRES  ", "MAP     "]),
            ("GRIDHEAD", 2 * np.ones((100,), dtype=np.int32)),
        ],
    )
    buf.seek(0)
    reader = xtge.EGridReader(buf)

    with pytest.raises(NotImplementedError, match="unstructured"):
        reader.read()


def test_read_no_endgrid():
    buf = io.BytesIO()
    eclio.write(
        buf,
        [
            ("FILEHEAD", np.zeros((100,), dtype=np.int32)),
            ("GRIDUNIT", ["METRES  ", "MAP     "]),
            ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
            ("ZCORN   ", np.ones((8,), dtype=np.int32)),
            ("COORD   ", np.ones((4,), dtype=np.int32)),
        ],
    )
    buf.seek(0)
    reader = xtge.EGridReader(buf)

    with pytest.raises(xtge.EGridFileFormatError, match="ENDGRID"):
        reader.read()


def test_read_unexpected_section():
    buf = io.BytesIO()
    eclio.write(
        buf,
        [
            ("FILEHEAD", np.zeros((100,), dtype=np.int32)),
            ("GRIDUNIT", ["METRES  ", "MAP     "]),
            ("GRIDHEAD", np.ones((100,), dtype=np.int32)),
            ("ZCORN   ", np.ones((8,), dtype=np.int32)),
            ("COORD   ", np.ones((4,), dtype=np.int32)),
            ("ENDGRID ", []),
            ("SECTION ", []),
        ],
    )
    buf.seek(0)
    reader = xtge.EGridReader(buf)

    with pytest.raises(
        xtge.EGridFileFormatError, match="subsection started with unexpected"
    ):
        reader.read()


@settings(max_examples=5)
@given(xtgeo_compatible_egrids())
def test_coarsening_warning(egrid):
    assume(egrid.global_grid.corsnum is not None)

    with pytest.warns(UserWarning, match="coarsen"):
        egrid.xtgeo_coord()


@given(xtgeo_compatible_egrids())
def test_local_coordsys_warning(egrid):
    assume(egrid.global_grid.coord_sys is not None)

    with pytest.warns(UserWarning, match="coordinate definition"):
        egrid.xtgeo_coord()


@settings(max_examples=5)
@given(
    xtgeo_compatible_egrids(
        lgrs=st.lists(lgr_sections(), min_size=1, max_size=1),
    )
)
def test_lgr_warning(egrid):
    assume(len(egrid.lgr_sections) > 0)

    with pytest.warns(UserWarning, match="LGR"):
        egrid.xtgeo_coord()


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(xtgeo_compatible_egrids())
def test_zero_numres_backwards_compatibility(tmp_path, egrid):
    egrid.to_file(tmp_path / "grid1.egrid")
    egrid.global_grid.grid_head.numres = 0
    egrid.to_file(tmp_path / "grid2.egrid")

    grid1 = xtg.grid_from_file(tmp_path / "grid1.egrid")
    with pytest.warns(UserWarning, match="EGrid file given with numres < 1"):
        grid2 = xtg.grid_from_file(tmp_path / "grid2.egrid")

    assert grid1._coordsv.tolist() == grid2._coordsv.tolist()
    assert grid1._zcornsv.tolist() == grid2._zcornsv.tolist()
    assert grid1._actnumsv.tolist() == grid2._actnumsv.tolist()
