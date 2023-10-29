import os
from dataclasses import dataclass

import hypothesis.strategies as st
import numpy as np
import pytest
import resfo
from hypothesis import given, settings

import xtgeo
from xtgeo.grid3d._ecl_inte_head import InteHead
from xtgeo.grid3d._ecl_logi_head import LogiHead
from xtgeo.grid3d._ecl_output_file import Simulator, TypeOfGrid, UnitSystem

from ..test_grid3d.grid_generator import xtgeo_grids


@dataclass
class OpmSetup:
    grid: xtgeo.Grid
    formatted: bool
    unit_system: UnitSystem
    poro: float = 0.3

    def run(self):
        deck = "TEST.DATA"
        with open(deck, "w") as fh:
            fh.write(self.deck_contents())
        os.system(f"flow {deck}")
        return "TEST"

    @property
    def restart_file(self):
        if self.formatted:
            return "TEST.FUNRST"
        else:
            return "TEST.UNRST"

    @property
    def init_file(self):
        if self.formatted:
            return "TEST.FINIT"
        else:
            return "TEST.INIT"

    def convert_pressure(self, value):
        if self.unit_system == UnitSystem.METRIC:
            # barsa
            return value
        if self.unit_system == UnitSystem.FIELD:
            # psia
            return 14.5 * value
        if self.unit_system == UnitSystem.LAB:
            # atma
            return 0.987 * value

    def convert_length(self, value):
        if self.unit_system == UnitSystem.METRIC:
            # metres
            return value
        if self.unit_system == UnitSystem.FIELD:
            # feet
            return 3.28 * value
        if self.unit_system == UnitSystem.LAB:
            # cm
            return 1000 * value

    def convert_density(self, value):
        if self.unit_system == UnitSystem.METRIC:
            # kg/m3
            return value
        if self.unit_system == UnitSystem.FIELD:
            # lb/ft3
            return 0.0624 * value
        if self.unit_system == UnitSystem.LAB:
            # gm/cc
            return 0.001 * value

    def convert_volume_ratio(self, value):
        if self.unit_system == UnitSystem.METRIC:
            # sm3/sm3
            return value
        if self.unit_system == UnitSystem.FIELD:
            # MSCF/STB
            return 0.0056 * value
        if self.unit_system == UnitSystem.LAB:
            # scc/scc
            return value

    def convert_time(self, value):
        if self.unit_system == UnitSystem.METRIC:
            # days
            return value
        if self.unit_system == UnitSystem.FIELD:
            # days
            return value
        if self.unit_system == UnitSystem.LAB:
            # hours
            return 24 * value

    def deck_contents(self):
        nx, ny, nz = self.grid.dimensions
        self.grid.to_file("TEST.GRDECL", fformat="grdecl")
        size = nx * ny * nz
        return f"""
RUNSPEC
UNIFOUT
{"FMTOUT" if self.formatted else ""}
{self.unit_system.name}
DIMENS
 {nx} {ny} {nz} /
START
   1 'JAN' 2000 /
OIL
WATER
WELLDIMS
 0 /
EQLDIMS
 /
TABDIMS
 1* 1* 30 30 /
GRID
INIT
INCLUDE
 'TEST.GRDECL' /
PORO
 {size}*{self.poro} /
PERMX
 {size}*150 /
PERMY
 {size}*150 /
PERMZ
 {size}*150 /
PROPS
SWOF
 0.1 0.0    1.0 0.0
 0.5 2.0E-6 0.1 0.0
 1.0 1.0E-5 0.0 0.0 /
PVTW
 {self.convert_pressure(270.0)} 1.0 {1.0/self.convert_pressure(1.0/5.0E-5)} 0.3 0.0 /
DENSITY
 {self.convert_density(860.0)}
 {self.convert_density(1030.0)}
 {self.convert_density(0.8)} /
PVTO
 {self.convert_volume_ratio(0.001)} {self.convert_pressure(15.0)}    1.0 0.1 /
 {self.convert_volume_ratio(1.6)}   {self.convert_pressure(5000.0)}  1.8 0.5
                                    {self.convert_pressure(9000.0)}  1.7 0.6/
 /

SOLUTION
EQUIL
 {self.convert_length(1000)}
 {self.convert_pressure(4800)}
 {self.convert_length(1000)}
 0
 {self.convert_length(1000)}
 0
 0
 0
 0
 /
SUMMARY
FOPR
SCHEDULE
RPTSCHED
 'PRES'/
RPTRST
 'BASIC=1' /
TSTEP
 {self.convert_time(366)} /
END
"""


unit_systems = st.sampled_from(UnitSystem)

opm_setups = st.builds(OpmSetup, xtgeo_grids, st.booleans(), unit_systems)


@pytest.mark.requires_opm
@pytest.mark.usefixtures("setup_tmpdir")
@settings(max_examples=5)
@given(opm_setups)
def test_restart_header_reading(case):
    case.run()
    file_name = case.restart_file
    inte_head = None
    logi_head = None
    for kw, val in resfo.read(file_name):
        if kw == "INTEHEAD":
            inte_head = InteHead(val)
        if kw == "LOGIHEAD":
            logi_head = LogiHead.from_file_values(val, simulator=inte_head.simulator)

    assert not logi_head.dual_porosity
    assert not logi_head.radial

    assert inte_head.simulator == Simulator.ECLIPSE_100
    assert inte_head.num_x == case.grid.ncol
    assert inte_head.num_y == case.grid.nrow
    assert inte_head.num_z == case.grid.nlay
    assert inte_head.num_active == case.grid.nactive
    assert inte_head.type_of_grid == TypeOfGrid.CORNER_POINT
    assert inte_head.unit_system == case.unit_system


@pytest.mark.requires_opm
@pytest.mark.usefixtures("setup_tmpdir")
@settings(max_examples=5)
@given(opm_setups)
def test_init_header_reading(case):
    case.run()
    file_name = case.init_file
    inte_head = None
    logi_head = None
    for kw, val in resfo.read(file_name):
        if kw == "INTEHEAD":
            inte_head = InteHead(val)
        if kw == "LOGIHEAD":
            logi_head = LogiHead.from_file_values(val, simulator=inte_head.simulator)

    assert not logi_head.dual_porosity
    assert not logi_head.radial

    assert inte_head.simulator == Simulator.ECLIPSE_100
    assert inte_head.num_x == case.grid.ncol
    assert inte_head.num_y == case.grid.nrow
    assert inte_head.num_z == case.grid.nlay
    assert inte_head.num_active == case.grid.nactive
    assert inte_head.type_of_grid == TypeOfGrid.CORNER_POINT


@pytest.mark.requires_opm
@pytest.mark.usefixtures("setup_tmpdir")
@settings(max_examples=5)
@given(opm_setups)
def test_init_props_reading(case):
    case.run()
    poro = xtgeo.gridproperty_from_file(
        case.init_file, fformat="init", name="PORO", grid=case.grid
    )

    np.testing.assert_allclose(poro.values, case.poro)
    assert poro.date == "20000101"


@pytest.mark.requires_opm
@pytest.mark.usefixtures("setup_tmpdir")
@settings(max_examples=5)
@given(opm_setups)
def test_restart_prop_reading(case):
    case.run()
    if case.formatted:
        fformat = "funrst"
    else:
        fformat = "unrst"

    pressure = xtgeo.gridproperty_from_file(
        case.restart_file, fformat=fformat, name="PRESSURE", date="last", grid=case.grid
    )

    assert pressure.name == "PRESSURE_20010101"
    assert pressure.date == "20010101"
