"""I/O of blocked well data"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from xtgeo.common.log import null_logger
from xtgeo.io.welldata._well_io import WellData

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike


logger = null_logger(__name__)


@dataclass(frozen=True)
class BlockedWellData(WellData):
    """Immutable data container for blocked well data.

    This dataclass extends WellData with grid cell indices, representing a well
    that has been intersected with a 3D grid. Each survey point is associated with
    a grid cell defined by I, J, K indices.

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

    i_index: npt.NDArray[np.float64] = field(default=None)  # type: ignore[arg-type]
    j_index: npt.NDArray[np.float64] = field(default=None)  # type: ignore[arg-type]
    k_index: npt.NDArray[np.float64] = field(default=None)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        """Validate blocked well data consistency."""
        # First call parent validation
        super().__post_init__()

        # Validate that index arrays are provided
        if self.i_index is None or self.j_index is None or self.k_index is None:
            raise ValueError(
                "BlockedWellData requires i_index, j_index, and k_index arrays"
            )

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
        fformat: str = "csv",
        **kwargs: Any,
    ) -> BlockedWellData:
        """Read blocked well data from file with format selection.

        Factory method that delegates to format-specific readers based on the
        fformat parameter.

        Args:
            filepath: Path to input file or memory stream
            fformat: File format. Supported formats:
                - "csv": Comma-separated values file
                - "rms_ascii": RMS ASCII well format
            **kwargs: Format-specific keyword arguments passed to the reader

        Returns:
            BlockedWellData object

        Raises:
            ValueError: If fformat is not supported

        Example:
            >>> # Read from CSV
            >>> well = BlockedWellData.from_file("data.csv", fformat="csv",
            ...                                   wellname="WELL-1")
        """
        fformat = fformat.lower()

        if fformat == "csv":
            return cls.from_csv(filepath=filepath, **kwargs)

        if fformat == "rms_ascii":
            return cls.from_rms_ascii(filepath=filepath, **kwargs)

        raise ValueError(
            f"Unsupported file format: '{fformat}'. "
            f"Supported formats: 'csv', 'rms_ascii'"
        )

    def to_file(
        self,
        filepath: FileLike,
        fformat: str = "csv",
        **kwargs: Any,
    ) -> None:
        """Write blocked well data to file with format selection.

        Factory method that delegates to format-specific writers based on the
        fformat parameter.

        Args:
            filepath: Path to output file or memory stream
            fformat: File format. Supported formats:
                - "csv": Comma-separated values file
                - "rms_ascii": RMS ASCII well format
            **kwargs: Format-specific keyword arguments passed to the writer

        Raises:
            ValueError: If fformat is not supported

        Example:
            >>> # Write to CSV
            >>> well.to_file("output.csv", fformat="csv")
        """
        fformat = fformat.lower()

        if fformat == "csv":
            self.to_csv(filepath=filepath, **kwargs)
            return

        if fformat == "rms_ascii":
            self.to_rms_ascii(filepath=filepath, **kwargs)
            return

        raise ValueError(
            f"Unsupported file format: '{fformat}'. "
            f"Supported formats: 'csv', 'rms_ascii'"
        )

    @classmethod
    def from_csv(  # type: ignore[override]
        cls,
        filepath: FileLike,
        wellname: str | None = None,
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
        i_indexname: str = "I_INDEX",
        j_indexname: str = "J_INDEX",
        k_indexname: str = "K_INDEX",
        wellname_col: str = "WELLNAME",
    ) -> BlockedWellData:
        """Read blocked well data from CSV file.

        The CSV file can contain data for multiple wells. This method extracts data
        for a specific well by filtering on the wellname column.

        Args:
            filepath: Path to CSV file or memory stream
            wellname: Name of the well to extract from the CSV file. If None, uses
                the first well found in the file.
            xname: Column name for X coordinates (default: "X_UTME")
            yname: Column name for Y coordinates (default: "Y_UTMN")
            zname: Column name for Z coordinates (default: "Z_TVDSS")
            i_col: Column name for I-indices (default: "I_INDEX")
            j_col: Column name for J-indices (default: "J_INDEX")
            k_col: Column name for K-indices (default: "K_INDEX")
            wellname_col: Column name for well name (default: "WELLNAME")

        Returns:
            BlockedWellData object

        Raises:
            ValueError: If the specified wellname is not found in the CSV file

        Example:
            >>> well = BlockedWellData.from_csv("blocked_well.csv", wellname="WELL-1")
            >>> print(f"Well: {well.name}, Records: {well.n_records}")
        """
        from xtgeo.io.welldata.fformats._csv import read_csv_blockedwell

        return read_csv_blockedwell(
            filepath=filepath,
            wellname=wellname,
            xname=xname,
            yname=yname,
            zname=zname,
            i_indexname=i_indexname,
            j_indexname=j_indexname,
            k_indexname=k_indexname,
            wellname_col=wellname_col,
        )

    def to_csv(  # type: ignore[override]
        self,
        filepath: FileLike,
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
        i_indexname: str = "I_INDEX",
        j_indexname: str = "J_INDEX",
        k_indexname: str = "K_INDEX",
        wellname_col: str = "WELLNAME",
        include_header: bool = True,
    ) -> None:
        """Write blocked well data to CSV file.

        Args:
            filepath: Output CSV file path or memory stream
            xname: Column name for X coordinates (default: "X_UTME")
            yname: Column name for Y coordinates (default: "Y_UTMN")
            zname: Column name for Z coordinates (default: "Z_TVDSS")
            i_col: Column name for I-indices (default: "I_INDEX")
            j_col: Column name for J-indices (default: "J_INDEX")
            k_col: Column name for K-indices (default: "K_INDEX")
            wellname_col: Column name for well name (default: "WELLNAME")
            include_header: Whether to include column headers (default: True)

        Example:
            >>> well.to_csv("output.csv")
        """
        from xtgeo.io.welldata.fformats._csv import write_csv_blockedwell

        write_csv_blockedwell(
            blocked_well=self,
            filepath=filepath,
            xname=xname,
            yname=yname,
            zname=zname,
            i_indexname=i_indexname,
            j_indexname=j_indexname,
            k_indexname=k_indexname,
            wellname_col=wellname_col,
            include_header=include_header,
        )

    @classmethod
    def from_rms_ascii(
        cls,
        filepath: FileLike,
    ) -> BlockedWellData:
        """Read blocked well data from RMS ASCII file or stream.

        Args:
            filepath: Path to RMS ASCII file or a file-like stream object

        Returns:
            BlockedWellData object

        Example:
            >>> well = BlockedWellData.from_rms_ascii("blocked_well.txt")
            >>> print(f"Well: {well.name}, Records: {well.n_records}")
        """
        from xtgeo.io.welldata.fformats._rms_ascii import read_rms_ascii_blockedwell

        return read_rms_ascii_blockedwell(filepath=filepath)

    def to_rms_ascii(
        self,
        filepath: FileLike,
        precision: int = 4,
    ) -> None:
        """Write blocked well data to RMS ASCII file or stream.

        Args:
            filepath: Output RMS ASCII file path or a file-like stream object
            precision: Number of decimal places for floats (default: 4)

        Example:
            >>> well.to_rms_ascii("output.txt")
        """
        from xtgeo.io.welldata.fformats._rms_ascii import write_rms_ascii_blockedwell

        write_rms_ascii_blockedwell(
            blocked_well=self,
            filepath=filepath,
            precision=precision,
        )
