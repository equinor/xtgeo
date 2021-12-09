import io
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import ecl_data_io as eclio
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

import xtgeo
import xtgeo.grid3d._find_gridprop_in_eclrun as xtg_im_ecl
from xtgeo.grid3d._ecl_inte_head import InteHead
from xtgeo.grid3d._ecl_logi_head import LogiHead
from xtgeo.grid3d._ecl_output_file import Phases
from xtgeo.grid3d._grdecl_format import match_keyword

from .grdecl_grid_generator import finites
from .grid_generator import xtgeo_grids


def test_set_remaining_saturations():
    assert xtg_im_ecl.remainder_saturations({"SWAT": 0.5, "SGAS": 0.5}) == {"SOIL": 0.0}
    assert xtg_im_ecl.remainder_saturations({"SOIL": 1.0}) == {"SWAT": 0.0, "SGAS": 0.0}
    assert xtg_im_ecl.remainder_saturations({"SOIL": 0.0, "SGAS": 0.0}) == {"SWAT": 1.0}
    assert xtg_im_ecl.remainder_saturations({"SOIL": 0.25}) == dict()
    assert (
        xtg_im_ecl.remainder_saturations({"SOIL": 0.25, "SWAT": 0.5, "SGAS": 0.25})
        == dict()
    )
    with pytest.raises(ValueError, match="Unknown saturation"):
        xtg_im_ecl.remainder_saturations({"MUD": 3.0})


def test_date_from_intehead():
    intehead = MagicMock()

    intehead.year = None
    intehead.month = 1
    intehead.day = 1

    assert xtg_im_ecl.date_from_intehead(intehead) is None

    intehead.year = 2001
    intehead.month = 12
    intehead.day = 31

    assert xtg_im_ecl.date_from_intehead(intehead) == 20011231


@dataclass
class MockEntry:
    keyword: str
    array: Any = field(default_factory=list)

    def read_keyword(self):
        return self.keyword

    def read_array(self):
        return self.array

    def read_length(self):
        return len(self.array)


def test_filter_lgr():
    props = iter(
        [
            MockEntry("LGR"),
            MockEntry("LGRPROP"),
            MockEntry("ENDLGR"),
            MockEntry("PROP"),
        ]
    )
    with pytest.warns(UserWarning, match="Found LGR"):
        assert list(xtg_im_ecl.filter_lgr(props)) == [MockEntry("PROP")]


def test_read_headers():
    intehead, logihead, _ = xtg_im_ecl.peek_headers(
        iter(
            [
                MockEntry("INTEHEAD", np.zeros(411, dtype=np.int32)),
                MockEntry("LOGIHEAD", np.zeros(128, dtype=bool)),
            ]
        )
    )

    assert intehead == InteHead(np.zeros(411, dtype=np.int32))
    assert logihead == LogiHead.from_file_values(
        np.zeros(128, dtype=bool), simulator=intehead.simulator
    )


def test_section_generator():
    sections = xtg_im_ecl.section_generator(
        iter(
            [
                MockEntry("SEQNUM"),
                MockEntry("PROP"),
                MockEntry("SEQNUM"),
                MockEntry("PROP2"),
            ]
        )
    )
    assert [entry.read_keyword() for section in sections for entry in section] == [
        "PROP",
        "PROP2",
    ]


def test_section_generator_missing_seqnum():
    with pytest.raises(ValueError, match="did not start with SEQNUM"):
        next(
            xtg_im_ecl.section_generator(
                iter(
                    [
                        MockEntry("PROP"),
                    ]
                )
            )
        )


def test_read_values():
    intehead = InteHead(np.full(411, 7, dtype=np.int32))
    assert (
        xtg_im_ecl.read_values(
            iter([MockEntry("SGAS", 0.5), MockEntry("SWAT", 0.25)]),
            intehead,
            ["SOIL"],
        )["SOIL"]
        == 0.25
    )
    with pytest.raises(ValueError, match="duplicate"):
        xtg_im_ecl.read_values(
            iter([MockEntry("SWAT", 0.5), MockEntry("SWAT", 0.25)]),
            intehead,
            ["SWAT"],
        )["SWAT"]


@pytest.mark.parametrize(
    "generator, keyword, phases, expected",
    [
        (
            iter([]),
            "SGAS",
            Phases.GAS,
            1.0,
        ),
        (
            iter([]),
            "SOIL",
            Phases.OIL,
            1.0,
        ),
        (
            iter([]),
            "SWAT",
            Phases.WATER,
            1.0,
        ),
        (
            iter([]),
            "SGAS",
            Phases.OIL_WATER,
            0.0,
        ),
        (
            iter([MockEntry("SWAT", 0.25)]),
            "SOIL",
            Phases.OIL_WATER,
            0.75,
        ),
        (
            iter([MockEntry("SGAS", 0.3)]),
            "SOIL",
            Phases.OIL_GAS,
            0.7,
        ),
        (
            iter([MockEntry("SWAT", 0.4)]),
            "SGAS",
            Phases.GAS_WATER,
            0.6,
        ),
        (
            iter([MockEntry("SWAT", 0.5), MockEntry("SGAS", 0.25)]),
            "SOIL",
            Phases.OIL_WATER_GAS,
            0.25,
        ),
    ],
)
def test_read_values_phases(generator, keyword, phases, expected):
    intehead = InteHead(np.full(411, phases.value, dtype=np.int32))
    assert (
        xtg_im_ecl.read_values(
            generator,
            intehead,
            [keyword],
        )[keyword]
        == expected
    )


@given(xtgeo_grids, finites)
def test_match_scalar(grid, value):
    vals = xtg_im_ecl.make_gridprop_values(value, grid, False)
    np.testing.assert_allclose(vals, value)
    assert vals.shape == grid.dimensions


@given(st.floats(allow_nan=False), st.integers(min_value=0, max_value=10))
def test_expand_scalar_values(val, size):
    assert np.array_equal(
        xtg_im_ecl.expand_scalar_values(val, size, False), np.array([val] * size)
    )
    assert np.array_equal(
        xtg_im_ecl.expand_scalar_values(val, size, True), np.array([val] * size * 2)
    )


def test_pick_dualporo_values():
    values = np.arange(0, 100)
    assert np.array_equal(
        xtg_im_ecl.pick_dualporo_values(values, [], 50, True),
        np.arange(50, 100),
    )
    assert np.array_equal(
        xtg_im_ecl.pick_dualporo_values(values, [], 50, False),
        np.arange(0, 50),
    )
    actind = np.arange(100, 110)
    assert np.array_equal(
        xtg_im_ecl.pick_dualporo_values(values, actind, 120, True), np.arange(90, 100)
    )
    assert np.array_equal(
        xtg_im_ecl.pick_dualporo_values(values, actind, 120, False), np.arange(0, 10)
    )


def test_match_dualporo():
    grid = xtgeo.create_box_grid((4, 3, 5))
    grid._dualporo = True
    grid._dualperm = True
    actnum = np.ones(grid.dimensions, dtype=np.int32)
    actnum[0, 0, 0] = 0
    actnum[0, 0, 1] = 1
    actnum[0, 0, 2] = 2
    actnum[0, 0, 3] = 3
    grid._dualactnum = xtgeo.GridProperty(values=actnum)
    grid._actnumsv = grid._dualactnum.values >= 1

    file_values = np.ones(np.prod(grid.dimensions) * 2 - 1, dtype=np.float32)

    matched_values = xtg_im_ecl.make_gridprop_values(file_values, grid, False)
    assert matched_values[0, 0, 0] is np.ma.masked
    assert matched_values[0, 0, 1] == 1.0
    assert matched_values[0, 0, 2] == 0.0
    assert matched_values[0, 0, 3] == 1.0

    matched_values = xtg_im_ecl.make_gridprop_values(file_values, grid, True)
    assert matched_values[0, 0, 0] is np.ma.masked
    assert matched_values[0, 0, 1] == 0.0
    assert matched_values[0, 0, 2] == 1.0
    assert matched_values[0, 0, 3] == 1.0


property_names = st.text(
    min_size=8, max_size=8, alphabet=st.characters(min_codepoint=40, max_codepoint=126)
)


def test_gridprop_ecl_run_empty_files():
    with pytest.raises(ValueError, match="Reached end of file"):
        xtgeo.gridproperty_from_file(
            io.BytesIO(),
            fformat="init",
            name="PROP",
            grid=xtgeo.create_box_grid((2, 2, 2)),
        )
    with pytest.raises(ValueError, match="Could not find"):
        xtgeo.gridproperty_from_file(
            io.BytesIO(),
            fformat="unrst",
            name="PROP",
            grid=xtgeo.create_box_grid((2, 2, 2)),
            date=19991231,
        )


@dataclass
class EclRun:
    grid: xtgeo.Grid
    start_date: date
    step_date: date
    property_name: str

    @property
    def xtgeo_start_date(self):
        return (
            self.start_date.day
            + self.start_date.month * 100
            + self.start_date.year * 10000
        )

    @property
    def xtgeo_step_date(self):
        return (
            self.step_date.day
            + self.step_date.month * 100
            + self.step_date.year * 10000
        )

    @property
    def init_intehead_array(self):
        intehead = np.zeros(411, dtype=np.int32)
        intehead[8] = self.grid._ncol
        intehead[9] = self.grid._nrow
        intehead[10] = self.grid._nlay
        intehead[14] = 7
        intehead[64] = self.start_date.day
        intehead[65] = self.start_date.month
        intehead[66] = self.start_date.year
        intehead[94] = 100
        return intehead

    @property
    def init_intehead(self):
        return InteHead(self.init_intehead_array)

    @property
    def unrst_intehead_array(self):
        intehead = np.zeros(411, dtype=np.int32)
        intehead[8] = self.grid._ncol
        intehead[9] = self.grid._nrow
        intehead[10] = self.grid._nlay
        intehead[14] = 7
        intehead[64] = self.step_date.day
        intehead[65] = self.step_date.month
        intehead[66] = self.step_date.year
        intehead[94] = 100
        return intehead

    @property
    def unrst_intehead(self):
        return InteHead(self.unrst_intehead_array)

    @property
    def logihead_array(self):
        logihead = np.zeros(128, dtype=bool)
        logihead[6] = self.grid.dualporo
        return logihead

    @property
    def logihead(self):
        return LogiHead.from_file_values(self.logihead_array)

    def write_files(self, dir_path):
        eclio.write(
            dir_path / self.unrst_file,
            [
                ("SEQNUM  ", np.zeros(1)),
                ("INTEHEAD", self.unrst_intehead_array),
                ("LOGIHEAD", self.logihead_array),
                (self.property_name, np.zeros(np.prod(self.grid.dimensions))),
            ],
        )

    @property
    def init_file(self):
        buf = io.BytesIO()
        eclio.write(
            buf,
            [
                ("INTEHEAD", self.init_intehead_array),
                ("LOGIHEAD", self.logihead_array),
                (self.property_name, np.zeros(np.prod(self.grid.dimensions))),
            ],
        )
        buf.seek(0)
        return buf

    @property
    def unrst_file(self):
        buf = io.BytesIO()
        eclio.write(
            buf,
            [
                ("SEQNUM  ", np.zeros(1)),
                ("INTEHEAD", self.unrst_intehead_array),
                ("LOGIHEAD", self.logihead_array),
                (self.property_name, np.zeros(np.prod(self.grid.dimensions))),
            ],
        )
        buf.seek(0)
        return buf


ecl_runs = st.builds(EclRun, xtgeo_grids, st.dates(), st.dates(), property_names)


@given(ecl_runs, property_names)
def test_gridprop_ecl_run_name_not_found(ecl_run, prop_name):
    assume(not match_keyword(ecl_run.property_name, prop_name))
    with pytest.raises(ValueError, match="Could not find property"):
        xtgeo.gridproperty_from_file(
            ecl_run.init_file,
            fformat="init",
            name=prop_name,
            grid=ecl_run.grid,
        )
    with pytest.raises(ValueError, match="Could not find property"):
        xtgeo.gridproperty_from_file(
            ecl_run.unrst_file,
            fformat="unrst",
            name=prop_name,
            grid=ecl_run.grid,
            date=ecl_run.xtgeo_step_date,
        )


@given(ecl_runs, st.booleans(), st.booleans())
def test_gridprop_unrst_date_correct(ecl_run, use_alterate_date_form, use_first):
    search_date = ecl_run.xtgeo_step_date
    if use_first:
        search_date = "first"
    if use_alterate_date_form:
        step_date = ecl_run.step_date
        search_date = f"{step_date.year:04}-{step_date.month:02}-{step_date.day:02}"

    gridprop = xtgeo.gridproperty_from_file(
        ecl_run.unrst_file,
        fformat="unrst",
        name=ecl_run.property_name,
        grid=ecl_run.grid,
        date=search_date,
    )

    assert gridprop.name == ecl_run.property_name + "_" + str(ecl_run.xtgeo_step_date)
    assert gridprop.date == ecl_run.xtgeo_step_date


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ecl_runs)
def test_gridprop_unrst_same_formatted(tmp_path, ecl_run):
    funrst = tmp_path / "file.FUNRST"
    eclio.write(funrst, eclio.read(ecl_run.unrst_file), eclio.Format.FORMATTED)

    ecl_run.unrst_file.seek(0)

    funrst_gridprop = xtgeo.gridproperty_from_file(
        funrst,
        fformat="funrst",
        name=ecl_run.property_name,
        grid=ecl_run.grid,
        date=ecl_run.xtgeo_step_date,
    )
    unrst_gridprop = xtgeo.gridproperty_from_file(
        ecl_run.unrst_file,
        fformat="unrst",
        name=ecl_run.property_name,
        grid=ecl_run.grid,
        date=ecl_run.xtgeo_step_date,
    )

    assert unrst_gridprop.name == funrst_gridprop.name
    assert np.array_equal(unrst_gridprop.values, funrst_gridprop.values)


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ecl_runs)
def test_gridprop_init_same_formatted(tmp_path, ecl_run):
    finit = tmp_path / "file.FUNRST"
    eclio.write(finit, eclio.read(ecl_run.init_file), eclio.Format.FORMATTED)

    ecl_run.init_file.seek(0)

    finit_gridprop = xtgeo.gridproperty_from_file(
        finit,
        fformat="finit",
        name=ecl_run.property_name,
        grid=ecl_run.grid,
    )
    init_gridprop = xtgeo.gridproperty_from_file(
        ecl_run.init_file,
        fformat="init",
        name=ecl_run.property_name,
        grid=ecl_run.grid,
    )

    assert init_gridprop.name == finit_gridprop.name
    assert np.array_equal(init_gridprop.values, finit_gridprop.values)
