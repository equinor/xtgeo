from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

TextLine = list[str]

_BASE10_NUMBER: re.Pattern[str] = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)


class TextParser:
    """
    Static light-weight methods for text parsing,
    to support reading of text files with headers, numerical data and comments.
    """

    class NumericValueErrorCode(StrEnum):
        OK = "ok"
        NOT_SINGLE_NUMBER = "not_single_number"
        NOT_BASE10 = "not_base10"
        NOT_FINITE = "not_finite"

    @staticmethod
    def iter_nonempty_lines(
        stream: Iterable[str],
    ) -> Iterator[TextLine]:
        """
        Lazily iterate over text lines, yielding lists of strings
        typically representing words and numbers.
        Filters out empty lines.
        """
        for line in stream:
            split_line = line.strip().split()

            if not split_line:
                continue

            yield split_line

    @staticmethod
    def is_comment(line: TextLine, comment_prefixes: list[str]) -> bool:
        """
        Check if a line is a comment line,
        starting with any of the comment prefixes.
        Requires that the line is non-empty, which is guaranteed when using
        iter_nonempty_lines.
        """
        return any(line[0].startswith(prefix) for prefix in comment_prefixes)

    @staticmethod
    def starts_with_prefix(line: TextLine, prefix: str) -> bool:
        """
        Check if line starts with a specific prefix, typically being a keyword
        indicating a specific section of the file and followed by
        a set of numeric values.
        Requires that the line is non-empty, which is guaranteed when using
        iter_nonempty_lines.
        """
        return line[0].startswith(prefix)

    @staticmethod
    def strip_surrounding_quotes(value_txt: str) -> str:
        """Remove one matching pair of surrounding double quotes, if present."""
        if (
            len(value_txt) >= 2
            and value_txt.startswith('"')
            and value_txt.endswith('"')
        ):
            return value_txt[1:-1]
        return value_txt

    @staticmethod
    def is_base10_number(value_txt: str) -> bool:
        """
        Base-10 number or not
            (base-10 numbers include integers, decimals and scientific notation)
        """
        return _BASE10_NUMBER.fullmatch(value_txt) is not None

    @staticmethod
    def is_single_finite_base10_value(
        value_line: TextLine,
    ) -> TextParser.NumericValueErrorCode:
        """
        - Single value or not
        - Finite value or not
            (e.g., not infinity or NaN)
        - Base10 number or not
            (base-10 numbers include integers, decimals and scientific notation)
        """
        if len(value_line) != 1:
            return TextParser.NumericValueErrorCode.NOT_SINGLE_NUMBER

        value_txt = TextParser.strip_surrounding_quotes(value_line[0])

        if not TextParser.is_base10_number(value_txt):
            return TextParser.NumericValueErrorCode.NOT_BASE10

        value = float(value_txt)
        if not math.isfinite(value):
            return TextParser.NumericValueErrorCode.NOT_FINITE

        return TextParser.NumericValueErrorCode.OK
