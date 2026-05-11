from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

# A constituent string, typically representing a word or a number
Token = str
TokenizedLine = list[Token]

_BASE10_NUMBER: re.Pattern[str] = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)


def iter_lines(
    stream: Iterable[str],
) -> Iterator[TokenizedLine]:
    """
    Lazily iterate over text lines, yielding lists of tokens (strings),
    typically representing words and numbers.
    Filters out empty lines.
    """
    for line in stream:
        split_line = line.strip().split()

        if not split_line:
            continue

        yield split_line


def iter_noncomment_lines(
    stream: Iterable[str], comment_prefixes: Sequence[str]
) -> Iterator[TokenizedLine]:
    """
    Lazily iterate over text lines, yielding lists of tokens (strings).
    Filters out empty lines and comment lines.
    """
    for line in iter_lines(stream):
        if is_comment(line, comment_prefixes):
            continue

        yield line


def is_comment(line: Sequence[Token], comment_prefixes: Sequence[str]) -> bool:
    """
    Check if a line is a comment line,
    namely starting with any of the comment prefixes.
    """
    if not line:
        return False
    return any(line[0].startswith(prefix) for prefix in comment_prefixes)


def contains_single_token(line: Sequence[Token]) -> bool:
    """Check if the line consists of a single token."""
    return len(line) == 1


def strip_surrounding_delimiters(token: Token, delimiter: str) -> Token:
    """Remove one matching pair of surrounding delimiters, if present."""
    if (
        delimiter
        and len(token) >= 2 * len(delimiter)
        and token.startswith(delimiter)
        and token.endswith(delimiter)
    ):
        return token[len(delimiter) : -len(delimiter)]
    return token


def is_base10_number(token: Token) -> bool:
    """
    Base-10 number or not
        (base-10 numbers include integers, decimals and scientific notation)
    """
    return _BASE10_NUMBER.fullmatch(token) is not None


def is_finite_number(token: Token) -> bool:
    """
    Finite number or not
        (finite numbers include base-10 numbers, but exclude infinities and NaNs)
    """
    try:
        float_value = float(token)
        return math.isfinite(float_value)
    except ValueError:
        return False
