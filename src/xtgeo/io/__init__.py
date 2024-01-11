# ruff: noqa: F401
"""XTGeo io module"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._file_wrapper import FileWrapper

    __all__ = ["FileWrapper"]
