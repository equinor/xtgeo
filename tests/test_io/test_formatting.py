import sys

import numpy as np
import pytest

from xtgeo.io._formatting import format_number, round_to_power_of_ten_if_close


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (42, "42"),
        (-9999, "-9999"),
        (0, "0"),
        (np.int64(42), "42"),
        (-9999.0, "-9999.0"),
        (1.0, "1.0"),
        (np.float64(1.25), "1.25"),
        (1e33, "1e+33"),
        (5e-324, "5e-324"),
        (10**1000, str(10**1000)),
    ],
)
def test_format_number_preserves_unconstrained_representation(
    value: int | float | np.integer | np.floating, expected: str
) -> None:
    assert format_number(value) == expected


@pytest.mark.parametrize(
    ("value", "width", "expected"),
    [
        (123.456789, 6, "123.46"),
        (1.234567890123456e100, 15, "1.23456789e+100"),
        (np.float64(-123.0), np.int64(6), "-123.0"),
    ],
)
def test_format_number_rounds_float_to_fit(
    value: float | np.floating, width: int | np.integer, expected: str
) -> None:
    assert format_number(value, max_characters=width) == expected


def test_format_number_skips_tokens_that_overflow_to_infinity() -> None:
    result = format_number(sys.float_info.max, max_characters=16)

    assert result == "1.79769313e+308"
    assert np.isfinite(float(result))


def test_format_number_raises_when_only_overflowing_token_fits() -> None:
    with pytest.raises(ValueError, match="Cannot format"):
        format_number(sys.float_info.max, max_characters=6)


def test_format_number_does_not_normalize_near_power_of_ten() -> None:
    assert format_number(9.999999999e32) == "9.999999999e+32"


@pytest.mark.parametrize("value", [True, np.bool_(False), "1.0", None])
def test_format_number_rejects_unsupported_values(value: object) -> None:
    with pytest.raises(TypeError, match="numeric scalar"):
        format_number(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_format_number_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        format_number(value)


@pytest.mark.parametrize("width", [True, np.bool_(False), 1.5, "3"])
def test_format_number_rejects_non_integer_width(width: object) -> None:
    with pytest.raises(TypeError, match="max_characters"):
        format_number(1.0, max_characters=width)  # type: ignore[arg-type]


@pytest.mark.parametrize("width", [0, -1, np.int64(-2)])
def test_format_number_rejects_non_positive_width(width: int | np.integer) -> None:
    with pytest.raises(ValueError, match="max_characters"):
        format_number(1.0, max_characters=width)


@pytest.mark.parametrize("value", [1.0, 10**1000])
def test_format_number_raises_when_value_cannot_fit(value: int | float) -> None:
    with pytest.raises(ValueError, match="Cannot format"):
        format_number(value, max_characters=1)


@pytest.mark.parametrize(
    ("value", "tolerance", "expected"),
    [
        (9.999999999e32, 1e-9, 1e33),
        (999.5, 1e-3, 1000.0),
        (999.5, 1e-4, 999.5),
        (-99.99999999, 1e-9, -100.0),
        (0, 1e-9, 0),
        (np.float64(9.999999999), 1e-9, 10.0),
        (np.int64(9), 0.2, 10.0),
    ],
)
def test_round_to_power_of_ten_if_close(
    value: int | float | np.integer | np.floating,
    tolerance: float,
    expected: int | float,
) -> None:
    assert (
        round_to_power_of_ten_if_close(value, relative_tolerance=tolerance) == expected
    )


def test_round_to_power_of_ten_if_close_leaves_huge_integer_unchanged() -> None:
    huge = 10**1000
    assert round_to_power_of_ten_if_close(huge) == huge


def test_round_to_power_of_ten_if_close_accepts_integer_tolerance() -> None:
    assert round_to_power_of_ten_if_close(1000.0, relative_tolerance=0) == 1000.0
    assert (
        round_to_power_of_ten_if_close(999.9999999, relative_tolerance=0) == 999.9999999
    )


@pytest.mark.parametrize("value", [10**16 + 1, 10**16 - 1, -(10**16 + 1)])
def test_zero_tolerance_does_not_round_inexact_large_integer(value: int) -> None:
    assert round_to_power_of_ten_if_close(value, relative_tolerance=0) == value


def test_zero_tolerance_rounds_exact_large_integer_power() -> None:
    assert round_to_power_of_ten_if_close(10**16, relative_tolerance=0) == 1e16


@pytest.mark.parametrize(
    "tolerance",
    [
        -1.0,
        1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        10**1000,
        -(10**1000),
    ],
)
def test_round_to_power_of_ten_if_close_rejects_invalid_tolerance(
    tolerance: int | float,
) -> None:
    with pytest.raises(ValueError, match="relative_tolerance"):
        round_to_power_of_ten_if_close(9.5, relative_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [True, np.bool_(False), "0.1"])
def test_round_to_power_of_ten_if_close_rejects_non_numeric_tolerance(
    tolerance: object,
) -> None:
    with pytest.raises(TypeError, match="relative_tolerance must be"):
        round_to_power_of_ten_if_close(
            9.5,
            relative_tolerance=tolerance,  # type: ignore[arg-type]
        )
