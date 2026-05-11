import io
from collections.abc import Sequence

import pytest

from xtgeo.io._text_parsing_utils import (
    TokenizedLine,
    contains_single_token,
    is_base10_number,
    is_comment,
    is_finite_number,
    iter_lines,
    iter_noncomment_lines,
    strip_surrounding_delimiters,
)


def test_iter_lines_strips_splits_and_skips_blank_lines() -> None:
    stream = io.StringIO("\n  KEY   1  2  \n\t\n  VALUE\t3\n")

    assert list(iter_lines(stream)) == [
        ["KEY", "1", "2"],
        ["VALUE", "3"],
    ]


def test_iter_noncomment_lines_skips_configured_comment_prefixes() -> None:
    stream = io.StringIO("\n  \n# comment\nVRTX 1 0.0 0.0 0.0 CNXYZ\nTRGL 1 2 3\n")

    result = list(iter_noncomment_lines(stream, ["#"]))

    assert result == [
        ["VRTX", "1", "0.0", "0.0", "0.0", "CNXYZ"],
        ["TRGL", "1", "2", "3"],
    ]


def test_iter_noncomment_lines_only_uses_configured_prefixes() -> None:
    stream = io.StringIO("! comment\n#POINTS\n3\nVALUE ! not a comment\n")

    result = list(iter_noncomment_lines(stream, ["!"]))

    assert result == [["#POINTS"], ["3"], ["VALUE", "!", "not", "a", "comment"]]


@pytest.mark.parametrize(
    "line, comment_prefixes, expected",
    [
        (["#", "comment"], ["#", "--"], True),
        (["--comment"], ["#", "--"], True),
        (["value", "#", "comment"], ["#", "--"], False),
        ([], ["#", "--"], False),
    ],
)
def test_is_comment(
    line: Sequence[str], comment_prefixes: Sequence[str], expected: bool
) -> None:
    assert is_comment(line, comment_prefixes) is expected


@pytest.mark.parametrize(
    "line, expected",
    [
        (["value"], True),
        (["value", "other"], False),
        ([], False),
    ],
)
def test_contains_single_token(line: TokenizedLine, expected: bool) -> None:
    assert contains_single_token(line) is expected


@pytest.mark.parametrize(
    "token",
    [
        "0",
        "-1",
        "+1",
        "1.",
        ".5",
        "1.5",
        "1e-3",
        "-1.5E+3",
    ],
)
def test_is_base10_number_accepts_supported_number_formats(token: str) -> None:
    assert is_base10_number(token) is True


@pytest.mark.parametrize(
    "token",
    [
        "",
        ".",
        "1,2",
        "nan",
        "inf",
        "0x10",
        "1_000",
        "1e",
    ],
)
def test_is_base10_number_rejects_unsupported_number_formats(token: str) -> None:
    assert is_base10_number(token) is False


@pytest.mark.parametrize(
    "token, delimiter, expected",
    [
        ('"1.25"', '"', "1.25"),
        ('""1.25""', '"', '"1.25"'),
        ('"1.25', '"', '"1.25'),
        ('1.25"', '"', '1.25"'),
        ("1.25", '"', "1.25"),
        ("--1.25--", "--", "1.25"),
        ("!1.25!", "!", "1.25"),
        ("$1.25$", "$", "1.25"),
        ("1.25", "", "1.25"),
    ],
)
def test_strip_surrounding_delimiters_removes_only_one_matching_pair(
    token: str, delimiter: str, expected: str
) -> None:
    assert strip_surrounding_delimiters(token, delimiter) == expected


@pytest.mark.parametrize(
    "token, expected",
    [
        ("1.25", True),
        ("1e309", False),
        ("nan", False),
        ("inf", False),
        ("not-a-number", False),
    ],
)
def test_is_finite_number(token: str, expected: bool) -> None:
    assert is_finite_number(token) is expected
