from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np

from ._ecl_output_file import Phases, Simulator, TypeOfGrid, UnitSystem


class InteHead:
    """Contains the values for the INTEHEAD array in ecl restart
    and init files.

    Output files from eclipse and opm flow will contain sections
    starting with keyword-array headers. One of these is the INTEHEAD
    keyword. The values in the array are integers, and the meaning
    of each index is described in the e.g. the OPM user manual (2021-04 rev_01
    section D.6-D.7).

    The length of the array is not specified, meaning some values are missing,
    the InteHead class creates a lookup for these values:

    >>> intehead = InteHead(np.array([0,1,2,3,4,5,6,7,8]))
    >>> intehead.num_x
    8
    >>> # The year field is missing in the input
    >>> intehead.year is None
    True
    """

    def __init__(self, values: np.ndarray[np.int_, Any]) -> None:
        """Create an InteHead from the corresponding array.

        Args:
            values: Array of values following the INTEHEAD keyword
                in an ECL restart or init file.
        """
        self.values = values

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InteHead):
            return False

        return np.array_equal(self.values, other.values)

    def __repr__(self) -> str:
        return f"InteHead(values={self.values})"

    def __str__(self) -> str:
        return self.__repr__()

    def _optional_index_lookup(self, index: int) -> int | None:
        """Looks up the value at the given index, returning None if out of bound.

        Args:
            index: The index in the value array to look up
            constructor: Constructor function to wrap non-None values in, defaults
                to identity.
        Returns:
            value at the index, None if out of bounds.
        """
        return self.values[index] if len(self.values) > index else None

    @property
    def unit_system(self) -> UnitSystem | None:
        """
        The unit system used in the file.
        """
        v = self._optional_index_lookup(2)
        return None if v is None else UnitSystem(v)

    @property
    def num_x(self) -> int | None:
        """The number of columns (x direction) of cells"""
        return self._optional_index_lookup(8)

    @property
    def num_y(self) -> int | None:
        """The number of rows (y direction) of cells"""
        return self._optional_index_lookup(9)

    @property
    def num_z(self) -> int | None:
        """The number of layers (z direction) of cells"""
        return self._optional_index_lookup(10)

    @property
    def num_active(self) -> int | None:
        """The number of active cells"""
        return self._optional_index_lookup(11)

    @property
    def phases(self) -> Phases | None:
        """The phase system used for simulation"""
        if any([ids in str(self.simulator) for ids in ["300", "INTERSECT"]]):
            # item 14 in E300 runs is number of tracers, not IPHS; assume oil/wat/gas
            # item 14 in INTERSECT is always(?) undef., not IPHS; assume oil/wat/gas
            return Phases.OIL_WATER_GAS
        v = self._optional_index_lookup(14)
        return None if v is None else Phases(v)

    @property
    def day(self) -> int | None:
        """The simulated time calendar day

        (e.g. 3rd of april 2018)

        """
        return self._optional_index_lookup(64)

    @property
    def month(self) -> int | None:
        """The simulated time calendar month

        4 for simulation being in month 4.

        """
        return self._optional_index_lookup(65)

    @property
    def year(self) -> int | None:
        """The simulated time calendar month

        e.g. 2018 for simulation being done in year 2018
        """
        return self._optional_index_lookup(66)

    @property
    def simulator(self) -> Simulator | int | None:
        """The simulator used for producing the run, or integer code if unknown"""
        s_code = self._optional_index_lookup(94)
        try:
            return Simulator(s_code)
        except ValueError:
            warnings.warn(f"Unknown simulator code {s_code}")
            return s_code

    @property
    def type_of_grid(self) -> TypeOfGrid | None:
        """The type of grid used in the simulation"""
        value = self._optional_index_lookup(13)
        return (
            None
            if value is None
            else TypeOfGrid.alternate_code(cast(Literal[0, 1, 2, 3], value))
        )
