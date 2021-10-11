import warnings

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

    def __init__(self, values: np.ndarray):
        """Create an InteHead from the corresponding array.

        Args:
            values: Array of values following the INTEHEAD keyword
                in an ECL restart or init file.
        """
        self.values = values

    def __eq__(self, other):
        if not isinstance(other, InteHead):
            return False

        return np.array_equal(self.values, other.values)

    def __repr__(self):
        return f"InteHead(values={self.values})"

    def __str__(self):
        return self.__repr__()

    def _optional_index_lookup(self, index, constructor=(lambda x: x)):
        """Looks up the value at the given index, returning None if out of bound.

        Args:
            index: The index in the value array to look up
            constructor: Constructor function to wrap non-None values in, defaults
                to identity.
        Returns:
            value at the index, None if out of bounds.
        """
        if len(self.values) > index:
            return constructor(self.values[index])
        return None

    @property
    def unit_system(self) -> UnitSystem:
        """
        The unit system used in the file.
        """
        return self._optional_index_lookup(2, UnitSystem)

    @property
    def num_x(self):
        """The number of columns (x direction) of cells"""
        return self._optional_index_lookup(8)

    @property
    def num_y(self):
        """The number of rows (y direction) of cells"""
        return self._optional_index_lookup(9)

    @property
    def num_z(self):
        """The number of layers (z direction) of cells"""
        return self._optional_index_lookup(10)

    @property
    def num_active(self):
        """The number of active cells"""
        return self._optional_index_lookup(11)

    @property
    def phases(self):
        """The phase system used for simulation"""
        return self._optional_index_lookup(14, Phases)

    @property
    def day(self):
        """The simulated time calendar day

        (e.g. 3rd of april 2018)

        """
        return self._optional_index_lookup(64)

    @property
    def month(self):
        """The simulated time calendar month

        4 for simulation being in month 4.

        """
        return self._optional_index_lookup(65)

    @property
    def year(self):
        """The simulated time calendar month

        e.g. 2018 for simulation being done in year 2018
        """
        return self._optional_index_lookup(66)

    @property
    def simulator(self) -> Simulator:
        """The simulator used for producing the run, or integer code if unknown"""
        s_code = self._optional_index_lookup(94)
        try:
            return Simulator(s_code)
        except ValueError:
            warnings.warn(f"Unknown simulator code {s_code}")
            return s_code

    @property
    def type_of_grid(self):
        """The type of grid used in the simulation"""
        return self._optional_index_lookup(13, TypeOfGrid.alternate_code)
