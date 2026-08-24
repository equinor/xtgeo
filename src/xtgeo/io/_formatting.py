"""Utilities for serializing numeric scalars in text-based file formats.

The functions in this module separate general decimal formatting from optional,
lossy normalization. Callers must explicitly request normalization before
formatting when a file format or sentinel-value policy requires it.

Examples:
    Preserve the shortest round-trippable representation of an ordinary value::

        >>> format_number(427391.726575)
        '427391.726575'

    Round a value only as much as necessary to fit a fixed-width text field::

        >>> format_number(1234567890.123456, max_characters=15)
        '1234567890.1235'

    Normalize a format-specific sentinel explicitly before serializing it::

        >>> dummy = round_to_power_of_ten_if_close(9.999999999999999e32)
        >>> dummy
        1e+33
        >>> format_number(dummy, max_characters=15)
        '1e+33'
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

NumericScalar: TypeAlias = int | float | np.integer | np.floating


def _normalize_finite_scalar(
    value: NumericScalar, *, param_name: str = "value"
) -> int | float:
    """Convert a supported finite numeric scalar to its built-in Python type.

    Args:
        value: The scalar to validate and normalize.
        param_name: Public parameter name to attribute error messages to.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{param_name} must be a numeric scalar, excluding bool")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{param_name} must be finite")
        return normalized

    raise TypeError(f"{param_name} must be an int, float, or NumPy numeric scalar")


def format_number(
    value: NumericScalar,
    *,
    max_characters: int | np.integer | None = None,
) -> str:
    """Format a finite numeric scalar within an optional character limit.

    Python and NumPy integer scalars are written exactly with :class:`str`.
    Python and NumPy floating-point scalars initially use :func:`repr`, which
    gives the shortest decimal representation that round-trips to the same
    Python float. If that representation is too long, general notation is
    attempted with progressively fewer significant digits, from 17 down to 1.
    The first representation that fits and parses back to a finite float is
    returned. NumPy floating types are converted to Python ``float`` first, so
    extended-precision types may lose precision.

    General notation chooses fixed-point or scientific notation according to
    the value. A ``.0`` suffix is added to fixed-point, integer-valued float
    output so that parsing the token preserves its floating-point nature. The
    function rounds numeric values to meet a width; it never truncates text.
    It also never performs domain-specific normalization, such as replacing a
    nearby value with an exact power of ten.

    Args:
        value: A Python or NumPy integer or floating-point scalar. Boolean and
            non-finite values are rejected.
        max_characters: Maximum token length, including any sign, decimal
            point, and exponent. ``None`` applies no limit. A supplied limit
            must be a positive Python or NumPy integer, excluding boolean.

    Returns:
        A finite base-10 token. If a width is supplied, the token is no longer
        than ``max_characters``.

    Raises:
        TypeError: If ``value`` is not a supported numeric scalar or
            ``max_characters`` is not an integer.
        ValueError: If ``value`` is non-finite, ``max_characters`` is not
            positive, or no permitted representation fits the requested width.

    Examples:
        Unconstrained floats use their shortest round-trippable representation::

            >>> format_number(123.456789)
            '123.456789'
            >>> format_number(np.float64(1.25))
            '1.25'

        A width limit reduces precision by rounding rather than truncating::

            >>> format_number(123.456789, max_characters=6)
            '123.46'

        Integer formatting remains exact, including for NumPy scalars::

            >>> format_number(np.int64(42))
            '42'

        Values are not implicitly rounded to nearby powers of ten::

            >>> format_number(9.999999999e32)
            '9.999999999e+32'

        Impossible widths and non-finite values are rejected::

            >>> format_number(1.0, max_characters=2)
            Traceback (most recent call last):
                ...
            ValueError: Cannot format 1.0 within 2 characters.

            >>> format_number(float("inf"))
            Traceback (most recent call last):
                ...
            ValueError: value must be finite
    """
    normalized = _normalize_finite_scalar(value)

    if isinstance(max_characters, (bool, np.bool_)) or (
        max_characters is not None and not isinstance(max_characters, (int, np.integer))
    ):
        raise TypeError("max_characters must be a positive integer or None")
    if max_characters is not None:
        max_characters = int(max_characters)
        if max_characters <= 0:
            raise ValueError("max_characters must be a positive integer")

    formatted = str(normalized) if isinstance(normalized, int) else repr(normalized)
    if max_characters is None or len(formatted) <= max_characters:
        return formatted

    if isinstance(normalized, int):
        raise ValueError(
            f"Cannot format {normalized} within {max_characters} characters."
        )

    # Seventeen significant decimal digits are sufficient to round-trip an
    # IEEE 754 binary64 value; reduce from there only as needed to fit the width.
    for precision in range(17, 0, -1):
        formatted = f"{normalized:.{precision}g}"
        if "." not in formatted and "e" not in formatted.lower():
            formatted += ".0"
        if len(formatted) <= max_characters and math.isfinite(float(formatted)):
            return formatted

    raise ValueError(f"Cannot format {normalized} within {max_characters} characters.")


def round_to_power_of_ten_if_close(
    value: NumericScalar,
    *,
    relative_tolerance: float = 1e-9,
) -> int | float:
    """Return the nearest signed power of ten when a value is sufficiently close.

    This is an explicitly lossy normalization for format-specific policies,
    such as producing a stable text token for a sentinel value. It is not part
    of :func:`format_number` because ordinary coordinates and data values must
    not be changed merely because they happen to lie near a power of ten.

    The candidate is the signed power of ten whose exponent is nearest to
    ``log10(abs(value))``. It is returned only when :func:`math.isclose` accepts
    the difference with ``relative_tolerance`` and an absolute tolerance of
    zero. With zero tolerance, integer equality is checked exactly to avoid
    binary64 rounding. Zero is returned unchanged. An integer too large to
    convert to a finite Python float is also returned unchanged, avoiding
    implicit loss of precision when no meaningful comparison can be made.

    Args:
        value: A finite Python or NumPy integer or floating-point scalar.
            Boolean values are rejected.
        relative_tolerance: Maximum relative difference accepted for
            normalization. It must be finite and satisfy
            ``0 <= relative_tolerance < 1``.

    Returns:
        The nearest signed power of ten as a float when it is within tolerance;
        otherwise, the normalized built-in Python ``int`` or ``float`` value.

    Raises:
        TypeError: If ``value`` is not a supported numeric scalar, or if
            ``relative_tolerance`` is not a real numeric scalar.
        ValueError: If either input is non-finite, or if the tolerance is
            outside the interval ``[0, 1)``.

    Examples:
        The default tolerance normalizes a binary approximation near ``1e33``::

            >>> round_to_power_of_ten_if_close(9.999999999e32)
            1e+33

        The tolerance controls whether a nearby value is changed::

            >>> round_to_power_of_ten_if_close(999.5, relative_tolerance=1e-3)
            1000.0
            >>> round_to_power_of_ten_if_close(999.5, relative_tolerance=1e-4)
            999.5

        Negative values use the corresponding signed power, while zero and
        unrepresentably large integers remain unchanged::

            >>> round_to_power_of_ten_if_close(-99.99999999)
            -100.0
            >>> round_to_power_of_ten_if_close(0)
            0
            >>> huge = 10**1000
            >>> round_to_power_of_ten_if_close(huge) == huge
            True
    """
    normalized = _normalize_finite_scalar(value)
    tolerance = _normalize_finite_scalar(
        relative_tolerance, param_name="relative_tolerance"
    )
    if not 0 <= tolerance < 1:
        raise ValueError("relative_tolerance must be finite and in the interval [0, 1)")
    tolerance = float(tolerance)

    if normalized == 0:
        return normalized

    try:
        float_value = float(normalized)
    except OverflowError:
        return normalized

    exponent = round(math.log10(abs(float_value)))
    nearest_power = math.copysign(10.0**exponent, float_value)
    if tolerance == 0.0 and isinstance(normalized, int):
        return nearest_power if normalized == int(nearest_power) else normalized
    if math.isclose(
        float_value,
        nearest_power,
        rel_tol=tolerance,
        abs_tol=0.0,
    ):
        return nearest_power

    return normalized
