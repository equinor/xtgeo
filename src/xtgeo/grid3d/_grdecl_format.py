import warnings
from contextlib import contextmanager

import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def split_line(line):
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


def split_line_no_string(line):
    """
    Same as split_line, but does not handle string literals, instead
    its quite a bit faster.
    """
    for w in line.split():
        if w.startswith("--"):
            return
        yield w


def until_space(string):
    """
    returns the given string until the first space.
    Similar to string.split(max_split=1)[0] except
    initial spaces are not ignored:
    >>> until_space(" hello")
    ''
    >>> until_space("hello world")
    'hello'

    """
    result = ""
    for w in string:
        if w.isspace():
            return result
        result += w
    return result


def match_keyword(kw1, kw2):
    """
    Perhaps surprisingly, the eclipse input format considers keywords
    as 8 character strings with space denoting end. So PORO, 'PORO ', and
    'PORO    ' are all considered the same keyword.

    >>> match_keyword("PORO", "PORO ")
    True
    >>> match_keyword("PORO", "PERM")
    False

    """
    return until_space(kw1) == until_space(kw2)


def interpret_token(val):
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
    grdecl_file,
    keywords,
    simple_keywords=[],
    max_len=None,
    ignore=IGNORE_ALL,
    strict=True,
):
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

    def read_grdecl(grdecl_stream):
        words = []
        keyword = None
        line_splitter = split_line

        line_no = 1
        line = grdecl_stream.readline()

        while line:
            if line is None:
                break

            if keyword is None:
                if max_len:
                    snubbed = line[0 : min(max_len, len(line))]
                else:
                    snubbed = line
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
