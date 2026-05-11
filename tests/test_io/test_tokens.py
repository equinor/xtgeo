import io
from collections.abc import Sequence

import pytest

from xtgeo.io._tokens import (
    TokenizedLine,
    contains_single_token,
    is_base10_number,
    is_comment,
    is_finite_number,
    iter_lines,
    iter_noncomment_lines,
    line_matches,
    strip_surrounding_delimiters,
)


def test_iter_lines_strips_splits_and_skips_blank_lines() -> None:
    stream = io.StringIO("\n  KEY   1  2  \n\t\n  VALUE\t3\n")

    assert list(iter_lines(stream)) == [
        ["KEY", "1", "2"],
        ["VALUE", "3"],
    ]


@pytest.mark.parametrize(
    "text, comment_prefixes, expected",
    [
        # Skips blank lines and configured comment prefixes
        (
            "\n  \n# comment\nVRTX 1 0.0 0.0 0.0 CNXYZ\nTRGL 1 2 3\n",
            ["#"],
            [
                ["VRTX", "1", "0.0", "0.0", "0.0", "CNXYZ"],
                ["TRGL", "1", "2", "3"],
            ],
        ),
        # Only the configured prefixes count; prefix-like tokens mid-line are kept
        (
            "! comment\n#POINTS\n3\nVALUE ! not a comment\n",
            ["!"],
            [["#POINTS"], ["3"], ["VALUE", "!", "not", "a", "comment"]],
        ),
    ],
)
def test_iter_noncomment_lines(
    text: str, comment_prefixes: Sequence[str], expected: list[list[str]]
) -> None:
    assert list(iter_noncomment_lines(io.StringIO(text), comment_prefixes)) == expected


@pytest.mark.parametrize(
    "line, comment_prefixes, expected",
    [
        (["#", "comment"], ["#", "--"], True),
        (["--comment"], ["#", "--"], True),
        (["value", "#", "comment"], ["#", "--"], False),
        ([], ["#", "--"], False),
        (["value"], [], False),
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
    "token, expected",
    [
        ("0", True),
        ("-1", True),
        ("+1", True),
        ("1.", True),
        (".5", True),
        ("1.5", True),
        ("1e-3", True),
        ("+1E3", True),
        ("-1.5E+3", True),
        ("", False),
        (".", False),
        ("1,2", False),
        ("nan", False),
        ("inf", False),
        ("0x10", False),
        ("1_000", False),
        ("1e", False),
        ("+", False),
        ("-", False),
        (" 1", False),
    ],
)
def test_is_base10_number(token: str, expected: bool) -> None:
    assert is_base10_number(token) is expected


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
        # Boundary: token length exactly equals 2 * len(delimiter)
        ('""', '"', ""),
        # Boundary: token shorter than 2 * len(delimiter),
        # even if it starts/ends with it
        ("--", "--", "--"),
        ("", '"', ""),
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
        ("3", True),
        ("1e309", False),
        ("nan", False),
        ("inf", False),
        ("-inf", False),
        ("not-a-number", False),
    ],
)
def test_is_finite_number(token: str, expected: bool) -> None:
    assert is_finite_number(token) is expected


@pytest.mark.parametrize(
    "line, reference, expected",
    [
        # Exact single-token match
        (["HEADER"], "HEADER", True),
        # Exact multi-token match
        (["TS", "Solid"], "TS Solid", True),
        # Reference with extra/irregular whitespace still matches
        (["TS", "Solid"], "  TS   Solid  ", True),
        # Case-sensitive: differing case does not match
        (["ts", "Solid"], "TS Solid", False),
        # Token mismatch
        (["TS", "Other"], "TS Solid", False),
        # Length mismatch: line shorter than reference
        (["TS"], "TS Solid", False),
        # Length mismatch: line longer than reference
        (["TS", "Solid", "Extra"], "TS Solid", False),
        # Empty line never matches a non-empty reference
        ([], "HEADER", False),
        # Empty reference never matches a non-empty line
        (["HEADER"], "", False),
        # Both empty: returns False (no reference to match)
        ([], "", False),
        # Reference consisting only of whitespace: behaves like empty reference
        (["HEADER"], "   ", False),
    ],
)
def test_line_matches(line: Sequence[str], reference: str, expected: bool) -> None:
    assert line_matches(line, reference) is expected
