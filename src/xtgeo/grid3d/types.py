"""
This file contains commen types used in xtgeo, keep it free some logic.
"""

from typing import Literal

METRIC = Literal[
    "euclid",
    "horizontal",
    "east west vertical",
    "north south vertical",
    "x projection",
    "y projection",
    "z projection",
]
