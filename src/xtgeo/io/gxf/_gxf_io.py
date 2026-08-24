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
    - The parser recognizes but ignores such keys and their values, hence
    they are not propagated through to xtgeo.
    - Missing or invalid extension values raise ValueError,
    since they indicate an ill-formed header.

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
- #SENSE (only the default value 1 is supported)
- #GTYPE (only the default value 0 is supported)
- #GRID (only uncompressed values are supported)

Keys required by the GXF format are:
- #POINTS
- #ROWS
- #GRID

Optional keys have defaults per the GXF spec; when missing, the default is used.
- Explicit defaults: `PTSEPARATION`, `RWSEPARATION`, `XORIGIN`, `YORIGIN`, `ROTATION`.
- Implicit absence defaults: `DUMMY=None`, `GTYPE=uncompressed`, `SENSE=1`.

#GRID values must be finite base-10 numeric values.
Compressed #GRID values declared by #GTYPE != 0 are intentionally unsupported and
raise ValueError. The absence of the #GTYPE key implies that the grid is treated
as uncompressed by default.
Undefined values at grid nodes are represented by a specified dummy value (#DUMMY);
- Masked/undefined grid values require that a dummy value is specified.
- When a dummy value is provided, values equal to the dummy value are treated
  as masked/undefined.
- When no dummy value is provided, all finite values are treated as valid.

Note that the GXF specification says nothing about the physical meaning of the data in
the #GRID section (elevation, rock property, etc.);
it is up to the user to interpret these values.
Nevertheless, a typical use is to let the #GRID data represent surface elevation.

The parser doesn't support all GXF keys, and it doesn't support all possible values
(on the next line) for the supported keys.
- supported keys with unsupported values raise `ValueError`
- unsupported or unknown keys are warned and skipped
- unsupported keys with valid value lines; unsupported values are rejected
This trade-off is to allow batch processing of GXF files when unsupported keys
are present, while rejecting malformed files.



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

! x maximum (not part of GXF specification, not propagated to xtgeo)
##XMAX
439101.301489

! y maximum (not part of GXF specification, not propagated to xtgeo)
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
import math
import warnings
from dataclasses import InitVar, dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import Self

from xtgeo.common.version import __version__ as xtgeo_version
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.io._formatting import format_number, round_to_power_of_ten_if_close
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


class GXFSerializationWarning(UserWarning):
    """Warn that GXF serialization cannot preserve the input data exactly.

    Promote this warning to an exception when mask-preserving output is required::

        >>> import warnings
        >>> from xtgeo.io.gxf import GXFSerializationWarning
        >>> warnings.simplefilter("error", GXFSerializationWarning)

    This warning is emitted during streaming of the grid, so an exception may
    leave a partially written output.
    """


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
    _mask_values_equal_to_dummy: InitVar[bool] = True

    def __post_init__(self, _mask_values_equal_to_dummy: bool) -> None:
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

        if (
            self.dummy is not None
            and not isinstance(self.dummy, int)
            and not math.isfinite(self.dummy)
        ):
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
        if self.dummy is not None and _mask_values_equal_to_dummy:
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
        """Parse an ASCII GXF stream into validated, immutable grid data.

        Args:
            stream: Text lines containing the complete GXF document.
            fileref_errmsg: Reference to file for error messages.

        Returns:
            Parsed GXF metadata and grid values as a ``GXFData`` instance.
        """

        scalar_values: dict[str, float | int] = {}
        grid_values: list[float] = []
        grid_mask: list[bool] = []
        dummy_decimal: Decimal | None = None
        dummy_float: float | None = None
        dummy_alias_warned = False
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

            if not is_single_token(line):
                raise ValueError(
                    f"In file {fileref_errmsg}: Malformed GXF key line "
                    f"'{line[0]}': expected a single key token, but found "
                    f"{len(line)} tokens."
                )

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
                        grid_value = float(token_value)
                        grid_values.append(grid_value)
                        is_dummy = False
                        if dummy_decimal is not None and grid_value == dummy_float:
                            is_dummy = Decimal(token_value) == dummy_decimal
                            if not is_dummy and not dummy_alias_warned:
                                # Example: #DUMMY 1.0000000000000001 and a single
                                # #GRID value 1.0 parse with the grid value unmasked,
                                # but both are stored as float64 1.0. Writing emits
                                # 1.0 for both and issues a warning,
                                # but rereading masks the grid value.
                                # This should happen extremely rarely.
                                # A potential alternative is to choose a collision-free
                                # dummy value. But that is currently blocked by
                                # https://github.com/equinor/xtgeo/issues/1684
                                warnings.warn(
                                    "A GXF grid value differs from #DUMMY in "
                                    "decimal form but becomes equal after float64 "
                                    "conversion. The exact decimal comparison "
                                    "preserves the correct mask, but both tokens "
                                    "are numerically indistinguishable: they are "
                                    "stored as the same float64 value, so the "
                                    "underlying numeric values alone cannot identify "
                                    "which token was #DUMMY. If the data is written "
                                    "to GXF, the unmasked value may serialize as "
                                    "#DUMMY; the writer will warn about such a "
                                    "collision, and reading that output will mask "
                                    "previously valid data.",
                                    UserWarning,
                                    stacklevel=3,
                                )
                                dummy_alias_warned = True
                        grid_mask.append(is_dummy)
                break

            if key == "SENSE":
                # SENSE controls both the orientation of the grid
                # and right-handedness/left-handedness of the coordinate system.
                # The GXF default is 1, and this parser only supports that
                # default representation.
                value_line = next(lines, None)
                if value_line is None or value_line[0].startswith("#"):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Missing value for key '#SENSE'."
                    )

                if (
                    len(value_line) != 1
                    or strip_surrounding_delimiters(value_line[0], '"') != "1"
                ):
                    raise ValueError(
                        f"In file {fileref_errmsg}: Invalid value for key '#SENSE'. "
                        "The only supported value is 1."
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
            if key == "DUMMY":
                dummy_decimal = Decimal(value_token)
                dummy_float = float(value_token)

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
        if points <= 0:
            raise ValueError(
                f"In file {fileref_errmsg}: Invalid value '{points}' for key "
                "'#POINTS'. Expected a strictly positive integer."
            )
        if rows <= 0:
            raise ValueError(
                f"In file {fileref_errmsg}: Invalid value '{rows}' for key "
                "'#ROWS'. Expected a strictly positive integer."
            )
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
            mask_2d = np.array(grid_mask, dtype=bool).reshape((rows, points))
            masked_values = np.ma.array(values_2d, mask=mask_2d)
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
            _mask_values_equal_to_dummy=False,
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

        Warns:
            GXFSerializationWarning:
            Callers can promote this warning to
            an exception with :func:`warnings.simplefilter`.
        """

        # User-defined keys (not part of GXF specification)
        # 1) Calculate the maximum X and Y coordinates, write as ##XMAX and ##YMAX
        #    These keys are per user's request, and should not be changed without
        #    internal discussion.
        x_max = self.xorigin + (self.points - 1) * self.ptseparation
        if not math.isfinite(x_max):
            raise ValueError(
                "Derived GXF metadata ##XMAX must be finite."
                " (##XMAX is the maximum X coordinate of the grid.) \n"
                " Non-finite value indicates invalid grid configuration."
            )
        x_max_formatted = format_number(x_max)

        y_max = self.yorigin + (self.rows - 1) * self.rwseparation
        if not math.isfinite(y_max):
            raise ValueError(
                "Derived GXF metadata ##YMAX must be finite."
                " (##YMAX is the maximum Y coordinate of the grid.) \n"
                " Non-finite value indicates invalid grid configuration."
            )
        y_max_formatted = format_number(y_max)

        wrapped_file = FileWrapper(file)
        wrapped_file.check_folder(raiseerror=OSError)

        # Values are written 5 per line, right-adjusted to a common column width.
        # This formatting is per user's request, and should not be changed without
        # internal discussion.
        # Tests should ensure that changes are caught.

        max_line_length = 80
        max_values_per_line = 5
        max_characters_per_value = (
            max_line_length - (max_values_per_line - 1)
        ) // max_values_per_line
        extra_spaces = max_line_length - (
            max_values_per_line * max_characters_per_value + (max_values_per_line - 1)
        )
        spaces_per_gap, remaining_spaces = divmod(extra_spaces, max_values_per_line - 1)
        gap_widths = [
            1 + spaces_per_gap + (index < remaining_spaces)
            for index in range(max_values_per_line - 1)
        ]
        dummy_formatted = (
            format_number(
                round_to_power_of_ten_if_close(self.dummy),
                max_characters=max_characters_per_value,
            )
            if self.dummy is not None
            else None
        )
        dummy_collision_warned = False

        with wrapped_file.get_text_stream_write(encoding=encoding) as stream:
            stream.write(
                f"! GXF file generated by xtgeo, version {xtgeo_version}\n"
                "! (https://github.com/equinor/xtgeo)\n\n"
            )
            stream.write("#POINTS\n")
            stream.write(f"{format_number(self.points)}\n")
            stream.write("\n")

            stream.write("#ROWS\n")
            stream.write(f"{format_number(self.rows)}\n")
            stream.write("\n")

            stream.write("#PTSEPARATION\n")
            stream.write(f"{format_number(self.ptseparation)}\n")
            stream.write("\n")

            stream.write("#RWSEPARATION\n")
            stream.write(f"{format_number(self.rwseparation)}\n")
            stream.write("\n")

            stream.write("#XORIGIN\n")
            stream.write(f"{format_number(self.xorigin)}\n")
            stream.write("\n")

            stream.write("#YORIGIN\n")
            stream.write(f"{format_number(self.yorigin)}\n")
            stream.write("\n")

            stream.write("#ROTATION\n")
            stream.write(f"{format_number(self.rotation)}\n")
            stream.write("\n")

            if self.dummy is not None:
                stream.write("#DUMMY\n")
                stream.write(f"{dummy_formatted}\n")
                stream.write("\n")

            # User-defined keys start with '##' (not part of GXF specification):
            stream.write(f"##XMAX\n{x_max_formatted}\n")
            stream.write("\n")
            stream.write(f"##YMAX\n{y_max_formatted}\n")
            stream.write("\n")

            stream.write("#GRID\n")

            # GXF requires lines of at most 80 characters. Rows may wrap, but each
            # row must start on a new line.
            for row in self.grid:
                for start in range(0, len(row), max_values_per_line):
                    chunk = []
                    for value in row[start : start + max_values_per_line]:
                        if np.ma.is_masked(value) and dummy_formatted is not None:
                            formatted = dummy_formatted
                        else:
                            formatted = format_number(
                                float(value),
                                max_characters=max_characters_per_value,
                            )
                            if (
                                formatted == dummy_formatted
                                and not dummy_collision_warned
                            ):
                                # Cases where a grid value and the #DUMMY value
                                # are different but are formatted to the same value are
                                # rare; only issuing a warning avoids a second grid pass
                                # to identify a collision-free dummy value
                                # and preserves streaming performance in normal cases.
                                # Note that automatically choosing a new dummy value
                                # in case of a collision would require a potentially
                                # #time-consuming pre-read of the grid.
                                # Moreover, https://github.com/equinor/xtgeo/issues/1684
                                # explains that such a dummy value is currently not
                                # handled properly.
                                warnings.warn(
                                    "An unmasked GXF grid value serializes to the "
                                    "same token as the #DUMMY value "
                                    f"({dummy_formatted}); reading the exported GXF "
                                    "file will mask that value.",
                                    GXFSerializationWarning,
                                    stacklevel=2,
                                )
                                dummy_collision_warned = True
                        chunk.append(formatted)
                    gaps = len(chunk) - 1
                    if gaps == 0:
                        line = chunk[0].rjust(max_characters_per_value)
                    else:
                        line = chunk[0].rjust(max_characters_per_value)
                        for token, gap_width in zip(chunk[1:], gap_widths[:gaps]):
                            line += " " * gap_width + token.rjust(
                                max_characters_per_value
                            )
                    stream.write(line + "\n")
