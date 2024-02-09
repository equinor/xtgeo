from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from xtgeo.common import null_logger

if TYPE_CHECKING:
    from collections.abc import Generator
    from io import TextIOWrapper

    from xtgeo.common.types import FileLike

logger = null_logger(__name__)


def split_line(line: str) -> Generator[str, None, None]:
    """
    split a keyword line inside a grdecl file. This splits the values of a
    'simple' keyword into tokens. ie.

    >>> list(split_line("3 1.0 3*4 PORO 3*INC 'HELLO WORLD  ' 3*'NAME'"))
    ['3', '1.0', '3*4', 'PORO', '3*INC', "'HELLO WORLD  '", "3*'NAME'"]

    note that we do not require string literals to have delimiting space at the
    end, but at the start. This is to be permissive at the end (as there is no
    formal requirement for spaces at end of string literals), but no space at
    the start of a string literal might indicate a repeating count.

    >>> list(split_line("3'hello world'4"))
    ["3'hello world'", '4']

    """
    value = ""
    inside_str = False
    for char in line:
        if char == "'":
            # Either the start or
            # the end of a string literal
            if inside_str:
                yield value + char
                value = ""
                inside_str = False
            else:
                inside_str = True
                value += char
        elif inside_str:
            # inside a string literal
            value += char
        elif value and value[-1] == "-" and char == "-":
            # a comment
            value = value[0:-1]
            break
        elif char.isspace():
            # delimiting space
            if value:
                yield value
                value = ""
        else:
            value += char
    if value:
        yield value


def split_line_no_string(line: str) -> Generator[str, None, None]:
    """
    Same as split_line, but does not handle string literals, instead
    its quite a bit faster.
    """
    for w in line.split():
        if w.startswith("--"):
            return
        yield w


def match_keyword(kw1: str, kw2: str) -> bool:
    """
    Perhaps surprisingly, the eclipse input format considers keywords
    as 8 character strings with space denoting end. So PORO, 'PORO ', and
    'PORO    ' are all considered the same keyword.

    Note that spaces may also occur inside e.g. tracer keywords, hence 'G1 F' vs 'G1 S'
    are different keywords.

    >>> match_keyword("PORO", "PORO ")
    True
    >>> match_keyword("PORO", "PERM")
    False
    >>> match_keyword("MORETHAN8LETTERS1)", "MORETHAN8LETTER2")
    True
    >>> match_keyword("G1 F", "G1 S")
    False

    """
    return kw1[0:8].rstrip() == kw2[0:8].rstrip()


def interpret_token(val: str) -> list[str]:
    """
    Interpret a eclipse token, tries to interpret the
    value in the following order:
    * string literal
    * keyword
    * repreated keyword
    * number

    If the token cannot be matched, we default to returning
    the uninterpreted token.

    >>> interpret_token("3")
    ['3']
    >>> interpret_token("1.0")
    ['1.0']
    >>> interpret_token("'hello'")
    ['hello']
    >>> interpret_token("PORO")
    ['PORO']
    >>> interpret_token("3PORO")
    ['3PORO']
    >>> interpret_token("3*PORO")
    ['PORO', 'PORO', 'PORO']
    >>> interpret_token("3*'PORO '")
    ['PORO ', 'PORO ', 'PORO ']
    >>> interpret_token("3'PORO '")
    ["3'PORO '"]

    """
    if val[0] == "'" and val[-1] == "'":
        # A string literal
        return [val[1:-1]]
    if val[0].isalpha():
        # A keyword
        return [val]
    if "*" in val:
        multiplicand, value = val.split("*")
        return interpret_token(value) * int(multiplicand)
    return [val]


IGNORE_ALL = None


@contextmanager
def open_grdecl(
    grdecl_file: FileLike,
    keywords: list[str],
    simple_keywords: list[str] | None = None,
    max_len: int | None = None,
    ignore: list[str] | None = IGNORE_ALL,
    strict: bool = True,
) -> Generator[Generator[tuple[str, list[str]], None, None], None, None]:
    """Generates tuples of keyword and values in records of a grdecl file.

    The format of the file must be that of the GRID section of a eclipse input
    DATA file.

    The records looked for must be "simple" ie.  start with the keyword, be
    followed by single word values and ended by a slash ('/').

    .. code-block:: none

        KEYWORD
        value value value /

    reading the above file with :code:`open_grdecl("filename.grdecl",
    keywords="KEYWORD")` will generate :code:`[("KEYWORD", ["value", "value",
    "value"])]`

    open_grdecl does not follow includes, obey skips, parse MESSAGE commands or
    make exception for groups and subrecords.

    Raises:
        ValueError: when end of file is reached without terminating a keyword,
            or the file contains an unrecognized (or ignored) keyword.

    Args:
        keywords (List[str]): Which keywords to look for, these are expected to
        be at the start of a line in the file  and the respective values
        following on subsequent lines separated by whitespace. Reading of a
        keyword is completed by a final '\'. See example above.

        simple_keywords (List[str]): Similar to keywords, but faster and
        cannot contain any string literals, such as the GRIDUNIT keyword
        which can be followed by the string literal 'METRES '.

        max_len (int): The maximum significant length of a keyword (Eclipse
        uses 8) ignore (List[str]): Keywords that have no associated data, and
        should be ignored, e.g. ECHO. Defaults to ignore all keywords that are
        not part of the results.

        ignore (List[str]): list of unmatched keywords to ignore, defaults to
        ignoring all unmatched keywords. Any keyword not ignored and not in
        the list of keywords looked for will give an error unless strict=False.
        Although a keyword is ignored, if it has trailing values on new lines
        those are interpreted as keywords, in order to ignore keywords with
        trailing values, use strict=False and filter warnings. Alternatively,
        add it to the list of expected keywords.

        strict (boolean): Whether unmatched keywords should raise an error or
        a warning.
    """

    if simple_keywords is None:
        simple_keywords = []

    def read_grdecl(
        grdecl_stream: TextIOWrapper,
    ) -> Generator[tuple[str, list[str]], None, None]:
        words: list[str] = []
        keyword = None
        line_splitter = split_line

        line_no = 1
        line = grdecl_stream.readline()

        while line:
            if line is None:
                break

            if keyword is None:
                snubbed = line[0 : min(max_len, len(line))] if max_len else line
                simple_matched_keywords = [
                    kw for kw in simple_keywords if match_keyword(kw, snubbed)
                ]
                matched_keywords = [kw for kw in keywords if match_keyword(kw, snubbed)]
                if matched_keywords or simple_matched_keywords:
                    if matched_keywords:
                        keyword = matched_keywords[0]
                        line_splitter = split_line
                    else:
                        keyword = simple_matched_keywords[0]
                        line_splitter = split_line_no_string
                    logger.debug("Keyword %s found on line %d", keyword, line_no)
                elif (
                    list(split_line(line))  # Not an empty line
                    and ignore is not IGNORE_ALL  # Not ignoring all
                    and not any(
                        match_keyword(snubbed, i) for i in ignore
                    )  # Not ignoring this
                ):
                    if strict:
                        raise ValueError(
                            f"Unrecognized keyword {repr(line)} on line {line_no}"
                        )
                    else:
                        warnings.warn(
                            f"Unrecognized keyword {repr(line)} on line {line_no}"
                        )

            else:
                for word in line_splitter(line):
                    if word == "/":
                        yield (keyword, words)
                        keyword = None
                        words = []
                        break
                    words += interpret_token(word)
            line = grdecl_stream.readline()
            line_no += 1

        if keyword is not None:
            raise ValueError(f"Reached end of stream while reading {keyword}")

    with open(grdecl_file, "r") as stream:
        yield read_grdecl(stream)
