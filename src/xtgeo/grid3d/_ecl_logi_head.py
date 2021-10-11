from dataclasses import dataclass
from typing import List, Optional

from ._ecl_output_file import Simulator


def lookup_optional_code(values, index):
    if len(values) <= index:
        return None
    return values[index]


@dataclass
class LogiHead:
    """Contains the values for the LOGIHEAD array in restart and init files.

    Output files from eclipse and opm flow will contain sections
    starting with keyword-array headers. One of these is the LOGIHEAD
    keyword. The values in the array are booleans, and the meaning
    of each index is described in the e.g. the OPM user manual (2021-04 rev_01
    section D.6-D.7).

    The length of the array is not specified, meaning some values are missing,
    the InteHead class creates a lookup for these values:

    Generally, the field describe whether an option has been enabled in the
    model, ie. if logihead.dual_porosity is True then the model uses the
    dual porosity feature.

    >>> logihead = LogiHead.from_file_values([True, True, False], Simulator.ECLIPSE_100)
    >>> logihead.dissolved_gas
    True
    >>> # Whether coal bed methane is used is missing
    >>> logihead.coal_bed_methane is None
    True

    """

    dissolved_gas: Optional[bool] = None
    vaporized_oil: Optional[bool] = None
    directional: Optional[bool] = None
    radial: Optional[bool] = None
    reversible: Optional[bool] = None
    hysterisis: Optional[bool] = None
    dual_porosity: Optional[bool] = None
    end_point_scaling: Optional[bool] = None
    directional_end_point_scaling: Optional[bool] = None
    reversible_end_point_scaling: Optional[bool] = None
    alternate_end_point_scaling: Optional[bool] = None
    miscible_displacement: Optional[bool] = None
    scale_water_pressure1: Optional[bool] = None
    scale_water_pressure2: Optional[bool] = None
    coal_bed_methane: Optional[bool] = None

    @classmethod
    def from_file_values(cls, values: List[bool], simulator: Simulator):
        """Construct a LogiHead from the array following the LOGIHEAD keyword
        Args:
            values: The iterable of boolean values following the LOGIHEAD keyword
            simulator: The meaning of each field is simulator dependent, so
                the simulator must be given.
        """
        if simulator == Simulator.ECLIPSE_100:
            # Weirdly, eclipse_100 outputs reversible and radial flags
            # in swapped order.
            indices = [0, 1, 2, 4, 3, 6, 14, 16, 17, 18, 19, 35, 55, 56, 127]
        else:
            indices = [0, 1, 2, 3, 4, 6, 14, 16, 17, 18, 19, 35, 55, 56, 127]

        return cls(*[lookup_optional_code(values, i) for i in indices])
