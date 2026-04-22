import io

import pytest

from xtgeo.io._text_parser import TextParser


def test_iter_nonempty_lines_strips_splits_and_skips_blank_lines() -> None:
    stream = io.StringIO("\n  KEY   1  2  \n\t\n  VALUE\t3\n")

    assert list(TextParser.iter_nonempty_lines(stream)) == [
        ["KEY", "1", "2"],
        ["VALUE", "3"],
    ]


@pytest.mark.parametrize(
    "line, comment_prefixes, expected",
    [
        (["#", "comment"], ["#", "--"], True),
        (["--comment"], ["#", "--"], True),
        (["value", "#", "comment"], ["#", "--"], False),
    ],
)
def test_is_comment(
    line: list[str], comment_prefixes: list[str], expected: bool
) -> None:
    assert TextParser.is_comment(line, comment_prefixes) is expected


@pytest.mark.parametrize(
    "line, prefix, expected",
    [
        (["KEYWORD", "1"], "KEY", True),
        (["KEY", "1"], "KEY", True),
        (["VALUE", "1"], "KEY", False),
    ],
)
def test_starts_with_prefix(line: list[str], prefix: str, expected: bool) -> None:
    assert TextParser.starts_with_prefix(line, prefix) is expected


@pytest.mark.parametrize(
    "value_txt",
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
def test_is_base10_number_accepts_supported_number_formats(value_txt: str) -> None:
    assert TextParser.is_base10_number(value_txt) is True


@pytest.mark.parametrize(
    "value_txt",
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
def test_is_base10_number_rejects_unsupported_number_formats(value_txt: str) -> None:
    assert TextParser.is_base10_number(value_txt) is False


@pytest.mark.parametrize(
    "value_txt, expected",
    [
        ('"1.25"', "1.25"),
        ('""1.25""', '"1.25"'),
        ('"1.25', '"1.25'),
        ('1.25"', '1.25"'),
        ("1.25", "1.25"),
    ],
)
def test_strip_surrounding_quotes_removes_only_one_matching_pair(
    value_txt: str, expected: str
) -> None:
    assert TextParser.strip_surrounding_quotes(value_txt) == expected


@pytest.mark.parametrize(
    "value_line, expected",
    [
        (["1.25"], TextParser.NumericValueErrorCode.OK),
        (['"1.25"'], TextParser.NumericValueErrorCode.OK),
        (['""1.25""'], TextParser.NumericValueErrorCode.NOT_BASE10),
        (['"1.25'], TextParser.NumericValueErrorCode.NOT_BASE10),
        (['1.25"'], TextParser.NumericValueErrorCode.NOT_BASE10),
        (["1", "2"], TextParser.NumericValueErrorCode.NOT_SINGLE_NUMBER),
        (["not-a-number"], TextParser.NumericValueErrorCode.NOT_BASE10),
        (["1e309"], TextParser.NumericValueErrorCode.NOT_FINITE),
    ],
)
def test_is_single_finite_base10_value(
    value_line: list[str], expected: TextParser.NumericValueErrorCode
) -> None:
    assert TextParser.is_single_finite_base10_value(value_line) == expected
