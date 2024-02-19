import hashlib

import pytest
from xtgeo.common.sys import generic_hash


def test_generic_hash():
    """Testing generic hashlib function."""
    ahash = generic_hash("ABCDEF")
    assert ahash == "8827a41122a5028b9808c7bf84b9fcf6"

    ahash = generic_hash("ABCDEF", hashmethod="sha256")
    assert ahash == "e9c0f8b575cbfcb42ab3b78ecc87efa3b011d9a5d10b09fa4e96f240bf6a82f5"

    ahash = generic_hash("ABCDEF", hashmethod="blake2b")
    assert ahash[0:12] == "0bb3eb1511cb"

    with pytest.raises(ValueError):
        ahash = generic_hash("ABCDEF", hashmethod="invalid")

    # pass a hashlib function
    ahash = generic_hash("ABCDEF", hashmethod=hashlib.sha224)
    assert ahash == "fd6639af1cc457b72148d78e90df45df4d344ca3b66fa44598148ce4"
