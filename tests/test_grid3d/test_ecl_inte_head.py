import numpy as np
import pytest

from xtgeo.grid3d._ecl_inte_head import InteHead, Phases
from xtgeo.grid3d._ecl_output_file import Simulator, TypeOfGrid, UnitSystem


def test_intehead_eq():
    intehead = InteHead(np.zeros(411, dtype=np.int32))

    assert intehead != ""
    assert intehead == InteHead(np.zeros(411, dtype=np.int32))
    assert intehead != InteHead(np.ones(411, dtype=np.int32))


def test_intehead_optional_lookup():
    intehead = InteHead(np.zeros(10, dtype=np.int32))

    assert intehead.num_active is None


def test_intehead_non_standard_simulator():
    intehead = InteHead(np.full(shape=411, fill_value=100, dtype=np.int32))

    assert intehead.simulator == Simulator.ECLIPSE_100


def test_intehead_iphs_when_e300():
    intehead_values = [0] * 100
    intehead_values[94] = 300  # simulator is Ecl 300
    intehead_values[14] = 8  # 14 is IPHS code in E100 but no. tracers in E300, here 8
    assert InteHead(intehead_values).phases == Phases.OIL_WATER_GAS, (
        "phases always OIL_WATER_GAS in Eclipse 300"
    )


def test_intehead_iphs_fail_when_outsiderange_e100():
    intehead_values = [0] * 100
    intehead_values[94] = 100  # simulator is Ecl 100
    intehead_values[14] = 8  # 14 is IPHS code in E100 but 8 is not a valid code
    with pytest.raises(ValueError, match="not a valid Phases"):
        InteHead(intehead_values).phases


def test_intehead_type_of_grid():
    intehead = InteHead(np.full(shape=411, fill_value=3, dtype=np.int32))

    assert intehead.type_of_grid == TypeOfGrid.alternate_code(3)


@pytest.mark.parametrize(
    "enum_member, code",
    [
        (TypeOfGrid.CORNER_POINT, 0),
        (TypeOfGrid.UNSTRUCTURED, 1),
        (TypeOfGrid.COMPOSITE, 2),
        (TypeOfGrid.BLOCK_CENTER, 3),
    ],
)
def test_type_of_grid_alternate_values(enum_member, code):
    assert enum_member.alternate_value == code
    assert TypeOfGrid.alternate_code(code) == enum_member


def test_intehead_unit_system():
    intehead = InteHead(np.full(shape=411, fill_value=3, dtype=np.int32))
    assert intehead.unit_system == UnitSystem(3)
