"""GXF file parser/writer for regular surfaces.

GXF (Grid eXchange Format) is a simple ASCII format for
a regular, rotated grid with values at each node (not cells).
This module provides functionality to read and write the GXF format,
but only a subset of the keys are supported.

Documentation of the GXF format:
https://pubs.usgs.gov/of/1999/of99-514/grids/gxf.pdf



The file typically consists of the following items:

! Comment line
    - Starts with '!'
    - Is ignored by the file reader.

#KEY (single #)
value for KEY (on the next line after KEY)
    - A KEY preceeded with a single "#" means that the KEY is part
    of the GXF specification
    - The GXF format does not allow value in quotes. However, some files
    have them and the parser will strip them away if present.
    - Keys that are not handled by the parser are ignored with a warning
    and skipped.

##KEY (double #)
value for KEY (on next line after KEY)
    - A KEY preceeded by '##' is NOT part
    of the GXF specification, but added by some software or user.
    The parser recognizes but ignores such keys and their values.

Free text
    - Any line that does NOT start with "!", "#" or "##" is considered
    free text and ignored by the file reader.
    - Exception: the #GRID key is followed by the #GRID section

About the #GRID section
    - Follows immediately after the #GRID key
    - Is required by the GXF specification
    - Is a set of values given at the grid nodes
    - All lines in the #GRID section must contain only values; keys are not allowed.
    - Comment lines starting with '!' are ignored.
    - Each line may be up to 80 characters long,
    but the number of values per line is not fixed.



The parser supports the following GXF keys:
    (xtgeo.RegularSurface counterpart in parentheses)

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

Optional keys have defaults per the GXF spec; when missing, the default is used.

Compressed #GRID values declared by #GTYPE != 0 are intentionally unsupported and
raise ValueError. The absence of #GTYPE is treated as uncompressed by default.

Undefined values at grid nodes are represented by a specified dummy value (#DUMMY);
when a dummy value is provided, values equal to the dummy value are treated
as masked/undefined. When no dummy value is provided, all finite values are treated
as valid.
#GRID values must be finite base-10 numeric values.
Masked/undefined grid values require that a dummy value is specified.

Note that the GXF specification says nothing about the physical meaning of the data in
the #GRID section (elevation, rock property, etc.);
it is up to the user to interpret these values.
Nevertheless, a typical use is to let the #GRID data represent surface elevation.




==================================================================================
Example of a valid GXF file:
==================================================================================

! This is a comment line, starting with '!'
! Existing examples include CRS data, but it is ignored by the parser.

Some free text

! number of points in x-direction (ncol)
#POINTS
3

! number of points in y-direction (nrow)
#ROWS
2

! point separation in x-direction (xinc)
#PTSEPARATION
30.0

! row separation in y-direction (yinc)
#RWSEPARATION
30.0

! x origin (xori)
#XORIGIN
427391.726575

! y origin (yori)
#YORIGIN
7250373.922731

! counterclockwise rotation in degrees (rotation)
#ROTATION
66.57993141719481

! value in undefined nodes (undef)
#DUMMY
9999999.0

! x maximum (not part of GXF specification)
##XMAX
439101.301489

! y maximum (not part of GXF specification)
##YMAX
7260149.299141

! grid values at the nodes
#GRID
9999999.0      9999999.0     9999999.0
3288.2225       2837.758      2844.5764

! The layout in the #GRID section is:
! 3 points along the x-axis (namely, 3 columns)
! 2 points along the y-axis (namely, 2 rows)

==================================================================================
End example of a valid GXF file
==================================================================================

"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import Self

from xtgeo.common.version import __version__ as xtgeo_version
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.io._tokens import (
    TokenizedLine,
    is_finite_decimal_number,
    is_single_token,
    iter_noncomment_lines,
    strip_surrounding_delimiters,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import NotImplementedType

    from xtgeo.common.types import FileLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class GXFData:
    """Internal immutable data representation for a regular surface in GXF format."""

    DEFAULTS: ClassVar[dict[str, float]] = {
        "PTSEPARATION": 1.0,
        "RWSEPARATION": 1.0,
        "XORIGIN": 0.0,
        "YORIGIN": 0.0,
        "ROTATION": 0.0,
    }

    points: int
    rows: int
    ptseparation: float
    rwseparation: float
    xorigin: float
    yorigin: float
    rotation: float
    dummy: int | float | None
    grid: np.ma.MaskedArray

    def __post_init__(self) -> None:
        values = np.ma.array(self.grid, copy=True)

        if self.points <= 0 or self.rows <= 0:
            raise ValueError("points and rows must be strictly positive integers.")

        for name, value in {
            "ptseparation": self.ptseparation,
            "rwseparation": self.rwseparation,
            "xorigin": self.xorigin,
            "yorigin": self.yorigin,
            "rotation": self.rotation,
        }.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if self.dummy is not None and not np.isfinite(self.dummy):
            raise ValueError("dummy value must be a finite number.")

        if values.shape != (self.rows, self.points):
            raise ValueError(
                "Invalid shape of values in the #GRID section. \n"
                f"Expected {(self.rows, self.points)}, but got {values.shape}."
            )

        if self.ptseparation <= 0.0:
            raise ValueError("ptseparation must be a strictly positive number.")

        if self.rwseparation == 0.0:
            raise ValueError("rwseparation must be a non-zero number.")

        # Ensure undefined nodes are represented by a mask when #DUMMY is defined.
        values = np.ma.masked_invalid(values)
        if self.dummy is not None:
            values = np.ma.masked_where(values == self.dummy, values)
        values.mask = np.ma.getmaskarray(values)
        if self.dummy is None and np.ma.count_masked(values):
            raise ValueError("Masked GXF grid values require a dummy value.")
        values.setflags(write=False)
        values._mask.setflags(write=False)  # type: ignore[attr-defined]
        object.__setattr__(self, "grid", values)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, GXFData):
            return NotImplemented

        if (
            self.points != other.points
            or self.rows != other.rows
            or self.ptseparation != other.ptseparation
            or self.rwseparation != other.rwseparation
            or self.xorigin != other.xorigin
            or self.yorigin != other.yorigin
            or self.rotation != other.rotation
            or self.dummy != other.dummy
        ):
            return False

        mask = np.ma.getmaskarray(self.grid)
        other_mask = np.ma.getmaskarray(other.grid)
        if not np.array_equal(mask, other_mask):
            return False

        return bool(np.array_equal(self.grid.data[~mask], other.grid.data[~other_mask]))

    def allclose(self, other: object, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        if not isinstance(other, GXFData):
            return False

        if self.points != other.points or self.rows != other.rows:
            return False

        for value, other_value in (
            (self.ptseparation, other.ptseparation),
            (self.rwseparation, other.rwseparation),
            (self.xorigin, other.xorigin),
            (self.yorigin, other.yorigin),
            (self.rotation, other.rotation),
        ):
            if not np.isclose(value, other_value, rtol=rtol, atol=atol):
                return False

        if self.dummy is None or other.dummy is None:
            if self.dummy is not other.dummy:
                return False
        elif not np.isclose(self.dummy, other.dummy, rtol=rtol, atol=atol):
            return False

        mask = np.ma.getmaskarray(self.grid)
        other_mask = np.ma.getmaskarray(other.grid)
        if not np.array_equal(mask, other_mask):
            return False

        return bool(np.ma.allclose(self.grid, other.grid, rtol=rtol, atol=atol))

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

    @staticmethod
    def _is_user_extension(line: TokenizedLine) -> bool:
        """
        Check if a line is a user extension (starts with '##').
        A KEY preceded by '##' is NOT part
        of the GXF specification, but added by some software or user.
        The parser recognizes but ignores such keys and their values.
        """
        return line[0].startswith("##")

    @classmethod
    def _parse_gxf(cls, stream: Iterable[str], fileref_errmsg: str) -> Self:

        scalar_values: dict[str, float | int] = {}
        grid_values: list[float] = []
        grid_found = False

        int_keys = {"POINTS", "ROWS", "GTYPE"}
        float_keys = {
            "PTSEPARATION",
            "RWSEPARATION",
            "XORIGIN",
            "YORIGIN",
            "ROTATION",
        }
        # DUMMY preserves the type from the input file.
        scalar_keys = int_keys | float_keys | {"DUMMY"}

        lines = iter_noncomment_lines(stream, ["!"])
        for line in lines:
            if not line[0].startswith("#"):
                # Free text outside the grid section:
                continue

            if line[0] != line[0].upper():
                raise ValueError(
                    f"In file {fileref_errmsg}: GXF keys must be uppercase; "
                    f"found '{line[0]}'."
                )

            if GXFData._is_user_extension(line):
                extension_key = line[0]
                # The value line is read and verified, but its value is ignored
                # since it is a user extension.
                value_line = next(lines, None)
                if value_line is None or value_line[0].startswith("#"):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Missing value for "
                        f"key '{extension_key}'.\n"
                        f"Ill-formed header is not accepted."
                    )
                if not is_single_token(value_line) or not is_finite_decimal_number(
                    strip_surrounding_delimiters(value_line[0], '"')
                ):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Invalid value for "
                        f"key '{extension_key}'. "
                        f"Expected a single finite decimal number."
                    )
                continue

            # Set the key and handle its value on the next line..
            # The #GRID key is handled separately since it is followed
            # by multiple value lines instead of a single value line as
            # for the other KEYs.
            key = line[0][1:]

            if key == "GRID":
                grid_found = True
                for grid_line in lines:
                    if grid_line[0].startswith("#"):
                        raise ValueError(
                            f"In file {fileref_errmsg}: Unexpected key "
                            f"'{''.join(grid_line)}' inside #GRID section."
                        )

                    for token in grid_line:
                        token_value = strip_surrounding_delimiters(token, '"')
                        if not is_finite_decimal_number(token_value):
                            raise ValueError(
                                f"In file {fileref_errmsg}: Invalid value "
                                f"'{token_value}' inside #GRID section. Only "
                                "finite decimal numbers are allowed."
                            )
                        grid_values.append(float(token_value))
                break

            if key == "SENSE":
                # SENSE controls both the orientation of the grid
                # and right-handedness/left-handedness of the coordinate system.
                # The current implementation does not take this parameter
                # properly into account, but emits an error since it impacts
                # the representation of the object.
                # The only supported value is 0, which corresponds to
                # the current implementation.
                value_line = next(lines, None)
                if value_line is None or value_line[0].startswith("#"):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Missing value for key '#SENSE'."
                    )

                if (
                    len(value_line) != 1
                    or strip_surrounding_delimiters(value_line[0], '"') != "0"
                ):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Invalid value for key '#SENSE'. "
                        "The only supported value is 0."
                    )
                continue

            if key not in scalar_keys:
                value_line = next(lines, None)
                if value_line is None or value_line[0].startswith("#"):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Missing value for ignored "
                        f"key '#{key}'.\n"
                        f"Ill-formed header is not accepted."
                    )
                msg = f"In file {fileref_errmsg}: Ignoring unknown GXF key '#{key}'."
                logger.warning(msg)
                warnings.warn(msg, UserWarning, stacklevel=3)
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

            value_token = strip_surrounding_delimiters(raw_value, '"')
            if not is_single_token(value_line) or not is_finite_decimal_number(
                value_token
            ):
                raise ValueError(
                    f"In file {fileref_errmsg}: Invalid value '{raw_value}' for "
                    f"key '#{key}'. Expected a single finite decimal number."
                )

            try:
                parsed_value: float | int
                if key in int_keys:
                    parsed_value = int(value_token)
                elif key == "DUMMY":
                    # Preserve the original type: int or float
                    try:
                        parsed_value = int(value_token)
                    except ValueError:
                        parsed_value = float(value_token)
                else:
                    parsed_value = float(value_token)
            except ValueError as err:
                raise ValueError(
                    f"In file {fileref_errmsg}: Invalid value '{raw_value}' for "
                    f"key '#{key}'."
                ) from err

            if key == "GTYPE" and parsed_value != 0:
                raise ValueError(
                    f"In file {fileref_errmsg}: Compressed GXF #GRID values "
                    "declared by '#GTYPE' are not supported."
                )

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
            # The GXF specification says the default is no dummy value.
            dummy_val = None
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
        """Read a GXF file.

        Args:
            file: Path to GXF file or a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the input file.
        """

        wrapped_file = FileWrapper(file)
        if not wrapped_file.check_file():
            raise FileNotFoundError(
                f"\nIn file {wrapped_file.name}:\nThe file does not exist."
            )

        # We let strict=False because the format allows a large number of
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

        # GXF spec requires lines <= 80 chars; rows may wrap but each new
        # row must start on a new line.
        max_line_length = 80

        with wrapped_file.get_text_stream_write(encoding=encoding) as stream:
            stream.write(
                f"! GXF file generated by xtgeo, version {xtgeo_version}\n"
                "! (https://github.com/equinor/xtgeo)\n\n"
            )
            stream.write("#POINTS\n")
            stream.write(f"{self._format_number(self.points)}\n")
            stream.write("\n")

            stream.write("#ROWS\n")
            stream.write(f"{self._format_number(self.rows)}\n")
            stream.write("\n")

            stream.write("#PTSEPARATION\n")
            stream.write(f"{self._format_number(self.ptseparation)}\n")
            stream.write("\n")

            stream.write("#RWSEPARATION\n")
            stream.write(f"{self._format_number(self.rwseparation)}\n")
            stream.write("\n")

            stream.write("#XORIGIN\n")
            stream.write(f"{self._format_number(self.xorigin)}\n")
            stream.write("\n")

            stream.write("#YORIGIN\n")
            stream.write(f"{self._format_number(self.yorigin)}\n")
            stream.write("\n")

            stream.write("#ROTATION\n")
            stream.write(f"{self._format_number(self.rotation)}\n")
            stream.write("\n")

            if self.dummy is not None:
                stream.write("#DUMMY\n")
                stream.write(f"{self._format_number(self.dummy)}\n")
                stream.write("\n")

            stream.write("#GRID\n")

            # Each line may be up to 80 characters long
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
                stream.write(current_line + "\n")
