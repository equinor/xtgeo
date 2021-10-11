import numpy as np
import pytest

from xtgeo.grid3d._ecl_inte_head import InteHead
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
