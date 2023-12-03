from unittest.mock import mock_open, patch

import pytest
from xtgeo.grid3d._grdecl_format import open_grdecl


@pytest.mark.parametrize(
    "file_data",
    [
        "PROP\n 1 2 3 4 / \n",
        "OTHERPROP\n 1 2 3 /\n 4 5 /\nPROP\n 1 2 3 4 / \n",
        "OTHERPROP\n 1 2 3 /\n 4 5 /\n/ PROP Eclipse comment\nPROP\n 1 2 3 4 / \n",
        "PROP\n 1 2 3 4 /",
        "PROP\n -- a comment \n 1 2 3 4 /",
        "-- a comment \nPROP\n \n 1 2 3 4 /",
        "PROP\n \n 1 2 \n -- a comment \n 3 4 /",
        "NOECHO\nPROP\n \n 1 2 \n -- a comment \n 3 4 /",
        "ECHO\nPROP\n \n 1 2 \n -- a comment \n 3 4 /",
        "NOECHO\nPROP\n \n 1 2 \n -- a comment \n 3 4 / \n ECHO",
    ],
)
def test_read_simple_property(file_data):
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            assert list(kw) == [("PROP", ["1", "2", "3", "4"])]


@pytest.mark.parametrize(
    "repeats, value",
    [(6, 1.0), (0, 1), (3, "INP")],
)
def test_read_repeated_property(repeats, value):
    inp_str = f"PROP\n {repeats}*{value} /\n"
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            assert list(kw) == [("PROP", [str(value)] * repeats)]


def test_read_repeated_string_literal():
    inp_str = "PROP\n 3*'INP   ' /\n"
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            assert list(kw) == [("PROP", ["INP   "] * 3)]


def test_read_string():
    inp_str = "PROP\n 'FOO BAR' FOO /\n"
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            assert list(kw) == [("PROP", ["FOO BAR", "FOO"])]


def test_read_extra_keyword_characters():
    file_data = (
        "LONGPROP Eclipse comment\n"
        "1 2 3 4 / More Eclipse comment\nOTHERPROP\n 5 6 7 8 /\n"
    )
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with open_grdecl(mock_file, keywords=["LONGPROP", "OTHERPROP"]) as kw:
            assert list(kw) == [
                ("LONGPROP", ["1", "2", "3", "4"]),
                ("OTHERPROP", ["5", "6", "7", "8"]),
            ]


def test_read_long_keyword():
    very_long_keyword = "a" * 200
    file_data = f"{very_long_keyword} Eclipse comment\n" "1 2 3 4 /"
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with open_grdecl(mock_file, keywords=[very_long_keyword]) as kw:
            assert list(kw) == [
                (very_long_keyword, ["1", "2", "3", "4"]),
            ]


@pytest.mark.parametrize(
    "undelimited_file_data",
    [
        "PROP\n 1 2 3 4 \n",
        "PROP\n 1 2 3 4 ECHO",
        "ECHO\nPROP\n 1 2 3 4",
        "PROP\n 1 2 3 4 -- a comment",
        "NOECHO\nPROP\n 1 2 3 4 -- a comment",
    ],
)
def test_read_prop_raises_error_when_no_forwardslash(undelimited_file_data):
    with patch(
        "builtins.open", mock_open(read_data=undelimited_file_data)
    ) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            with pytest.raises(ValueError):
                list(kw)
