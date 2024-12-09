import numpy as np

from xtgeo.grid3d._ecl_logi_head import LogiHead
from xtgeo.grid3d._ecl_output_file import Simulator


def test_logihead_optional_value():
    logihead = LogiHead.from_file_values(
        np.zeros(10, dtype=bool), simulator=Simulator.ECLIPSE_100
    )

    assert logihead.coal_bed_methane is None


def test_logihead_eclipse_100_specifics():
    values = np.zeros(10, dtype=bool)
    values[3] = True

    logihead100 = LogiHead.from_file_values(values, simulator=Simulator.ECLIPSE_100)
    logihead300 = LogiHead.from_file_values(values, simulator=Simulator.ECLIPSE_300)

    assert not logihead100.radial
    assert logihead100.reversible

    assert logihead300.radial
    assert not logihead300.reversible
