"""
Utility functions for parsing text-based (geoscience) data files,
typically containing text and numbers in structured formats.
- lazy iteration through lines from text file
- tokenizing text lines into lists of tokens (strings)
- filtering out comment lines
- identification of format-specific file sections (header, etc)
- verification of base-10 numbers (integers, decimals and scientific notation)
- verification of finite numbers (base-10 numbers, excluding infinities and NaNs)
"""

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


def line_matches(line: Sequence[Token], reference: str) -> bool:
    """
    Check if a tokenized line matches a reference string token-for-token.
    - Case-sensitive
    - Ignores leading/trailing whitespace and multiple spaces between tokens
    """
    if not line or not reference:
        return False

    tokens_to_match = reference.split()
    if len(line) != len(tokens_to_match):
        return False

    return all(token == ref_token for token, ref_token in zip(line, tokens_to_match))


def is_single_token(line: Sequence[Token]) -> bool:
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
        - base-10 numbers include integers, decimals and scientific notation
        - base-10 numbers include arbitrarily large numbers (e.g. "1e309")
        - base-10 numbers exclude infinities (inf, -inf) and NaNs
        - base-10 numbers exclude HEX numbers (e.g. "0xFF")
          and binary numbers (e.g. "0b1010")
        - base-10 numbers exclude numbers with underscores (e.g. "1_000"):
          Python's float() function accepts these, but they are not
          strictly base-10 numbers
    """
    return _BASE10_NUMBER.fullmatch(token) is not None


def is_finite_number(token: Token) -> bool:
    """
    Finite number or not
        - finite numbers include integers, decimals and scientific notation
        - finite numbers exclude arbitrarily large numbers (e.g. "1e309")
        - finite numbers exclude infinities (inf, -inf) and NaNs
        - finite numbers include HEX numbers (e.g. "0xFF")
          and binary numbers (e.g. "0b1010")
        - finite numbers include numbers with underscores (e.g. "1_000")
    """

    try:
        float_value = float(token)
        return math.isfinite(float_value)
    except ValueError:
        return False


def is_finite_decimal_number(token: Token) -> bool:
    """
    Finite decimal number or not
    ("everyday" numbers excluding infinities, NaNs, HEX, binary and very large numbers)
        - finite decimal numbers include integers, decimals and scientific notation
        - finite decimal numbers exclude infinities (inf, -inf) and NaNs
        - finite decimal numbers exclude very large numbers (e.g. "1e309")
        - finite decimal numbers exclude HEX numbers (e.g. "0xFF")
          and binary numbers (e.g. "0b1010")
        - finite decimal numbers exclude numbers with underscores (e.g. "1_000"):
          Python's float() function accepts these, but they are not
          strictly base-10 numbers
    """
    return is_base10_number(token) and is_finite_number(token)
