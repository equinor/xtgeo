"""I/O of blocked well data"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from xtgeo.common.log import null_logger
from xtgeo.io._welldata._well_io import WellData, WellFileFormat

if TYPE_CHECKING:  # pragma: no cover
    from xtgeo.common.types import FileLike

logger = null_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class BlockedWellData(WellData):
    """Immutable data container for blocked well data.

    This dataclass extends WellData with grid cell indices, representing a well
    that has been intersected with a 3D grid.

    This class is derived from WellData, but a difference for BlockedWell is that
    each survey point is associated with a grid cell location defined by
    I, J, K indices.

    Attributes:
        name: Well name (inherited from WellData)
        xpos: X-coordinate of the well header position (inherited)
        ypos: Y-coordinate of the well header position (inherited)
        zpos: Z-coordinate of the well header position (inherited)
        survey_x: X-coordinates along wellbore trajectory (inherited)
        survey_y: Y-coordinates along wellbore trajectory (inherited)
        survey_z: Z-coordinates along wellbore trajectory (inherited)
        logs: Tuple of WellLog objects (inherited)
        i_index: NumPy array of I-indices (column) for grid cells
        j_index: NumPy array of J-indices (row) for grid cells
        k_index: NumPy array of K-indices (layer) for grid cells

    Notes:
        - Grid indices should use 1-based indexing (typical in reservoir modeling)
        - Undefined/inactive cells can be represented with np.nan or a negative value
        - All index arrays must have the same length as survey arrays
        - Index arrays are typically integer-valued but stored as float to allow NaN
    """

    i_index: npt.NDArray[np.float64]
    j_index: npt.NDArray[np.float64]
    k_index: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate blocked well data consistency."""
        # First call parent validation
        super().__post_init__()

        # Check that index arrays have consistent lengths
        n_survey = len(self.survey_x)
        if len(self.i_index) != n_survey:
            raise ValueError(
                f"i_index has {len(self.i_index)} values, "
                f"but survey has {n_survey} points"
            )
        if len(self.j_index) != n_survey:
            raise ValueError(
                f"j_index has {len(self.j_index)} values, "
                f"but survey has {n_survey} points"
            )
        if len(self.k_index) != n_survey:
            raise ValueError(
                f"k_index has {len(self.k_index)} values, "
                f"but survey has {n_survey} points"
            )

    @property
    def has_valid_indices(self) -> bool:
        """Check if any grid indices are defined (not all NaN)."""
        return (
            not np.all(np.isnan(self.i_index))
            or not np.all(np.isnan(self.j_index))
            or not np.all(np.isnan(self.k_index))
        )

    @property
    def n_blocked_cells(self) -> int:
        """Return the number of cells with valid (non-NaN) grid indices."""
        # A cell is considered blocked if all three indices are valid
        valid_mask = (
            ~np.isnan(self.i_index) & ~np.isnan(self.j_index) & ~np.isnan(self.k_index)
        )
        return int(np.sum(valid_mask))

    def get_cell_indices(self, index: int) -> tuple[float, float, float]:
        """Get the grid cell indices at a specific survey point.

        Args:
            index: Survey point index (0-based)

        Returns:
            Tuple of (i, j, k) indices

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self.n_records:
            raise IndexError(
                f"Index {index} out of bounds for well with {self.n_records} records"
            )
        return (self.i_index[index], self.j_index[index], self.k_index[index])

    @classmethod
    def from_file(
        cls,
        filepath: FileLike,
        fformat: WellFileFormat = WellFileFormat.RMS_ASCII,
        **kwargs: Any,
    ) -> BlockedWellData:
        """Read blocked well data from file with format selection."""
        if fformat == WellFileFormat.RMS_ASCII:
            return cls.from_rms_ascii(filepath, **kwargs)

        raise NotImplementedError(f"File format {fformat} not supported yet.")

    def to_file(
        self,
        filepath: FileLike,
        fformat: WellFileFormat = WellFileFormat.RMS_ASCII,
        **kwargs: Any,
    ) -> None:
        """Write blocked well data to file with format selection."""
        if fformat == WellFileFormat.RMS_ASCII:
            self.to_rms_ascii(filepath, **kwargs)
            return

        raise NotImplementedError(f"File format {fformat} not supported yet.")

    @classmethod
    def from_rms_ascii(cls, filepath: FileLike) -> BlockedWellData:
        """Read blocked well data from RMS ASCII file."""
        from xtgeo.io._welldata._fformats._rms_ascii import read_rms_ascii_blockedwell

        return read_rms_ascii_blockedwell(filepath=filepath)

    def to_rms_ascii(
        self,
        filepath: FileLike,
        *,
        precision: int = 4,
    ) -> None:
        """Write blocked well data to RMS ASCII file."""
        from xtgeo.io._welldata._fformats._rms_ascii import write_rms_ascii_blockedwell

        write_rms_ascii_blockedwell(
            blocked_well=self,
            filepath=filepath,
            precision=precision,
        )
