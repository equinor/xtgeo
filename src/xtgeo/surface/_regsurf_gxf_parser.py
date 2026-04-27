"""GXF file parser/writer for regular surfaces.

GXF (Grid eXchange Format) is a simple ASCII format for regular grid data.
This module provides functionality to read and write a subset of the GXF format,
only a subset of the keys are supported.

Documentation: https://pubs.usgs.gov/of/1999/of99-514/grids/gxf.pdf



The file consists of the following items:

! comments starts with '!'
    - Comment: ignored by the file reader.

Free text
    - Comment: any line that does NOT start with '!' or '#' or "##" is considered
    free text and ignored by the file reader.

#KEY
value for KEY (on next line after KEY)
    - Comment: a KEY preceeded with a single '#' means that the KEY is part
    of the GFX specification
    - Comment: the GFX format does not allow value in quotes, but some files
    have them, they are stripped away if present.

##KEY (double #)
value for KEY (on next line after KEY)
    - Comment: a KEY preceeded by '##' means that the KEY is NOT part
    of the GFX specification, but added by some software or user.
    We ignore such keys, but log a warning.




Supported keys are: (xtgeo.RegularSurface counterpart)

- #POINTS (ncol)
- #ROWS (nrow)
- #PTSEPARATION (xinc)
- #RWSEPARATION (yinc)
- #XORIGIN (xori)
- #YORIGIN (yori)
- #ROTATION (rotation)
- #DUMMY (undef)
- #GRID (values)

Keys required by the GXF format are:
- #POINTS
- #ROWS
- #GRID

Optional keys have defaults per the GXF spec; when missing, the default is used and
a warning is issued.

The data represent a regular surface defined on a grid of ncol by nrow points,
with specified spacing (xinc, yinc), origin (xori, yori), and rotation.
The #GRID section contains values given at the grid nodes in row-major order
(rows of length ncol).
Undefined values at grid nodes are represented by a specified dummy value (#DUMMY);
when a dummy value is provided, values equal to the dummy value are treated
as masked/undefined. When no dummy value is provided, all values are treated as valid.
The GXF specification says nothing about the physical meaning of the data in
the #GRID section (elevation, rock property, etc.);
it is up to the user to interpret these values.

The parser will only pass on the values to the application, physical meaning is outside
the scope of this module.
Nevertheless, a typical use is to let the #GRID data represent surface elevation.


Lines starting with ``!`` are comments and ignored. Lines that do not start with
``#`` are considered free text and ignored (outside the ``#GRID`` section).

Keys prefixed with ``##`` are treated as extension keys and ignored with a warning.



==================================================================================
Example of a valid GXF file:
==================================================================================
! This is a comment line, starting with '!'

Some free text

#POINTS
"330"                     << number of columns (ncol, x-direction)
#ROWS
"208"                     << number of rows (nrow, y-direction)
#PTSEPARATION
"30.4800600"              << point separation in x-direction (xinc)
#RWSEPARATION
"30.4800600"              << row separation in y-direction (yinc)
#XORIGIN
"427391.726575"           << x origin (xori)
#YORIGIN
"7250373.922731"          << y origin (yori)
#ROTATION
"66.57993141719481"       << counterclockwise rotation in degrees (rotation)
#DUMMY
"9999999.0"               << value in undefined nodes
##XMAX
"439101.301489"           << x maximum (not part of GXF specification)
##YMAX
"7260149.299141"          << y maximum (not part of GXF specification)
#GRID
9999999.0      9999999.0     9999999.0         9999999.0         9999999.0
9999999.0      9999999.0     9999999.0         9999999.0         9999999.0
9999999.0      9999999.0     9999999.0         9999999.0         9999999.0
3288.2225       2837.758      2844.5764      2850.4048       2828.643
3289.9615      2830.3948      2813.4902      2812.1143      2810.7473
3253.7393        2793.37      2790.6665      2789.2434      2789.8645
... (more values in #GRID section, total of ncol*nrow values, in rows of length ncol)

It's layout thus is:
330 points along the x-axis (namely, 330 columns)
208 points along the y-axis (namely, 208 rows)

"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import Self

from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.io._text_parser import TextParser

if TYPE_CHECKING:
    from typing import TextIO

    from xtgeo.common.types import FileLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GXFData:
    """Internal immutable data representation for a regular surface in GXF format."""

    DEFAULTS: ClassVar[dict[str, float]] = {
        "PTSEPARATION": 1.0,
        "RWSEPARATION": 1.0,
        "XORIGIN": 0.0,
        "YORIGIN": 0.0,
        "ROTATION": 0.0,
        "DUMMY": 9999999.0,
    }

    points: int
    rows: int
    ptseparation: float
    rwseparation: float
    xorigin: float
    yorigin: float
    rotation: float
    dummy: int | float
    grid: np.ma.MaskedArray

    def __post_init__(self) -> None:
        values = self.grid

        if self.points <= 0 or self.rows <= 0:
            raise ValueError("points and rows must be positive integers.")

        if values.shape != (self.rows, self.points):
            raise ValueError(
                "Invalid shape of surface values in the #GRID section. \n"
                f"Expected {(self.rows, self.points)}, but got {values.shape}."
            )

        if self.ptseparation <= 0.0 or self.rwseparation <= 0.0:
            raise ValueError("xinc and yinc must be positive numbers.")

        # Ensure undefined nodes are represented by a mask.
        # masked_equal returns a new array, so no separate copy is needed.
        if values.mask is np.ma.nomask:
            values = np.ma.masked_equal(values, self.dummy)
            object.__setattr__(self, "grid", values)

    @staticmethod
    def _format_number(value: float | int) -> str:
        if isinstance(value, int):
            return str(value)
        formatted = f"{value:.16g}"
        # Ensure float values always contain a decimal point so that
        # the type is preserved on re-read (e.g. -9999.0 -> "-9999.0").
        if "." not in formatted and "e" not in formatted and "E" not in formatted:
            formatted += ".0"
        return formatted

    @classmethod
    def _parse_gxf(cls, stream: TextIO, fileref_errmsg: str) -> Self:

        scalar_values: dict[str, float | int] = {}
        grid_values: list[float] = []
        grid_found = False

        int_keys = {"POINTS", "ROWS"}
        float_keys = {
            "PTSEPARATION",
            "RWSEPARATION",
            "XORIGIN",
            "YORIGIN",
            "ROTATION",
        }
        # DUMMY preserves the type from the input file.
        scalar_keys = int_keys | float_keys | {"DUMMY"}

        lines = TextParser.iter_nonempty_lines(stream)
        for line in lines:
            if TextParser.is_comment(line, ["!"]):
                continue

            if not TextParser.starts_with_prefix(line, "#"):
                # Free text outside the grid section:
                # All keys start with '#' or '##'
                # Key values are handled when keys are processed
                continue

            if TextParser.starts_with_prefix(line, "##"):
                # Skip the value line for this extension key.
                next(lines, None)
                continue

            # Set the key and handle its value on the next line..
            # The #GRID key is handled separately since it is followed
            # by multiple value lines.
            key = line[0][1:].upper()

            if key == "GRID":
                grid_found = True
                for grid_line in lines:
                    if TextParser.is_comment(grid_line, ["!"]):
                        continue

                    if TextParser.starts_with_prefix(grid_line, "#"):
                        raise ValueError(
                            f"In file {fileref_errmsg}: Unexpected key "
                            f"'{''.join(grid_line)}' inside #GRID section."
                        )

                    for token in grid_line:
                        try:
                            grid_values.append(float(token))
                        except ValueError as err:
                            raise ValueError(
                                f"In file {fileref_errmsg}: Invalid value '{token}' "
                                "inside #GRID section."
                            ) from err
                break

            if key == "SENSE":
                # SENSE controls both the orientation
                # of the grid and right-handedness/left-handedness
                # of the coordinate system.
                # The current implementation does not take this parameter
                # into account.
                # But emit a warning since it impacts the surface representation.
                next(lines, None)  # skip value line
                msg = (
                    f"In file {fileref_errmsg}: Ignoring '#SENSE' key. "
                    "This key may affect grid orientation but is not "
                    "supported; verify that the surface orientation "
                    "and handedness is correct."
                )
                logger.warning(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)
                continue

            if key not in scalar_keys:
                next(lines, None)  # skip value line
                continue

            if key in scalar_values:
                raise ValueError(
                    f"In file {fileref_errmsg}: Duplicate key '#{key}' is not allowed."
                )

            value_line = next(lines, None)
            if value_line is None:
                raise ValueError(
                    f"In file {fileref_errmsg}: Missing value for key '#{key}'."
                )

            raw_value = value_line[0]
            if raw_value.startswith("#"):
                raise ValueError(
                    f"In file {fileref_errmsg}: Missing value for key '#{key}'."
                )

            # Strip quotes around the value if present
            value_txt = raw_value.strip('"')
            try:
                parsed_value: float | int
                if key in int_keys:
                    parsed_value = int(value_txt)
                elif key == "DUMMY":
                    # Preserve the original type: int or float
                    try:
                        parsed_value = int(value_txt)
                    except ValueError:
                        parsed_value = float(value_txt)
                else:
                    parsed_value = float(value_txt)
            except ValueError as err:
                raise ValueError(
                    f"In file {fileref_errmsg}: Invalid value '{raw_value}' for "
                    f"key '#{key}'."
                ) from err

            scalar_values[key] = parsed_value

        # Check for required keys
        required = ["POINTS", "ROWS"]
        missing_required = [k for k in required if k not in scalar_values]
        if missing_required:
            raise ValueError(
                f"In file {fileref_errmsg}: Missing mandatory keys: {missing_required}."
            )

        # #GRID is also required
        if not grid_found:
            raise ValueError(
                f"In file {fileref_errmsg}: Missing mandatory key '#GRID'."
            )

        # Apply defaults for optional keys
        for dkey, dval in cls.DEFAULTS.items():
            if dkey not in scalar_values:
                scalar_values[dkey] = dval

        points = int(scalar_values["POINTS"])
        rows = int(scalar_values["ROWS"])
        num_expected_values = points * rows
        if len(grid_values) != num_expected_values:
            raise ValueError(
                f"In file {fileref_errmsg}: Number of values in #GRID section "
                f"is {len(grid_values)}, but expected {num_expected_values} "
                f"(points*rows = {points}*{rows})."
            )

        values_2d = np.array(grid_values, dtype=np.float64).reshape((rows, points))

        if "DUMMY" in scalar_values:
            dummy_val = scalar_values["DUMMY"]
            masked_values = np.ma.masked_equal(values_2d, dummy_val)
        else:
            # internal sentinel; no actual masking since all values are valid.
            # So just to satisfy the dataclass field
            dummy_val = 1e33
            masked_values = np.ma.array(values_2d)

        return cls(
            points=points,
            rows=rows,
            ptseparation=float(scalar_values["PTSEPARATION"]),
            rwseparation=float(scalar_values["RWSEPARATION"]),
            xorigin=float(scalar_values["XORIGIN"]),
            yorigin=float(scalar_values["YORIGIN"]),
            rotation=float(scalar_values["ROTATION"]),
            dummy=dummy_val,
            grid=masked_values,
        )

    @classmethod
    def from_file(
        cls,
        file: FileLike,
        encoding: str = "utf-8",
    ) -> Self:
        """Read and parse a GXF file.

        Args:
            file: Path to GXF file or a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the input file.
        """

        wrapped_file = FileWrapper(file)
        if not wrapped_file.check_file():
            raise FileNotFoundError(
                f"\nIn file {wrapped_file.name}:\nThe file does not exist."
            )

        # We let 'strict=False' because the format allows a large number of
        # commented lines and free text lines at the beginning of the file.
        # Setting 'strict=True' implies reading the beginning of the file
        # into a buffer of limited size and checking for known format keys,
        # which fails if size of comments and free text exceeds the buffer size.
        # If this in fact not a GXF file, the reader will fail with
        # a more specific error message from the parsing logic.
        wrapped_file.fileformat(FileFormat.GXF.value[0], strict=False)

        with wrapped_file.get_text_stream_read(encoding=encoding) as stream:
            return cls._parse_gxf(stream, fileref_errmsg=str(wrapped_file.name))

    def to_file(
        self,
        file: FileLike,
        encoding: str = "utf-8",
    ) -> None:
        """Write GXFData to a file-like target in GXF format.

        Args:
            file: Path to GXF file or a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the output file.
        """

        wrapped_file = FileWrapper(file)
        wrapped_file.check_folder(raiseerror=OSError)

        with wrapped_file.get_text_stream_write(encoding=encoding) as stream:
            stream.write(
                "! GXF file generated by xtgeo (https://github.com/equinor/xtgeo)\n\n"
            )
            stream.write("#POINTS\n")
            stream.write(f'"{self._format_number(self.points)}"\n')

            stream.write("#ROWS\n")
            stream.write(f'"{self._format_number(self.rows)}"\n')

            stream.write("#PTSEPARATION\n")
            stream.write(f'"{self._format_number(self.ptseparation)}"\n')

            stream.write("#RWSEPARATION\n")
            stream.write(f'"{self._format_number(self.rwseparation)}"\n')

            stream.write("#XORIGIN\n")
            stream.write(f'"{self._format_number(self.xorigin)}"\n')

            stream.write("#YORIGIN\n")
            stream.write(f'"{self._format_number(self.yorigin)}"\n')

            stream.write("#ROTATION\n")
            stream.write(f'"{self._format_number(self.rotation)}"\n')

            stream.write("#DUMMY\n")
            stream.write(f'"{self._format_number(self.dummy)}"\n\n')

            stream.write("#GRID\n")

            # Write rows of ncol values from internal (ncol, nrow) representation.
            # GXF spec requires lines <= 80 chars; rows may wrap but each new
            # row must start on a new line.
            max_line_length = 80

            values_for_write = np.ma.filled(self.grid, fill_value=self.dummy)
            for row in values_for_write:
                tokens = [self._format_number(float(v)) for v in row]
                current_line = ""
                for token in tokens:
                    candidate = (current_line + " " + token) if current_line else token
                    if len(candidate) > max_line_length and current_line:
                        stream.write(current_line + "\n")
                        current_line = token
                    else:
                        current_line = candidate
                if current_line:
                    stream.write(current_line + "\n")
