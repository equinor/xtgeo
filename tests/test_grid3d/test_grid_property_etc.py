# coding: utf-8
"""Testing: test_grid_property more special functions"""

import pytest
from xtgeo.grid3d import _gridprop_roxapi


def test_rox_compatible_codes():
    """Test a local function '_rox_compatible_codes' in _gridprop_roxapi.py."""
    codes = _gridprop_roxapi._rox_compatible_codes({"1": "FOO", 2: "BAR"})
    assert codes == {1: "FOO", 2: "BAR"}

    codes = _gridprop_roxapi._rox_compatible_codes({"1": "FOO", None: "BAR"})
    assert codes == {1: "FOO"}

    with pytest.raises(ValueError, match="The keys in codes must be an integer"):
        codes = _gridprop_roxapi._rox_compatible_codes({"a": "FOO", 2: "BAR"})
