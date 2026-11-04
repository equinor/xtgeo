"""Tests for the public xtgeo RegularSurface <-> ResInsight API.

Covers ``xtgeo.regular_surface_from_resinsight`` and
``xtgeo.RegularSurface.to_resinsight``. A live ResInsight instance is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.interfaces.resinsight._rips_package import rips

pytestmark = [
    pytest.mark.requires_resinsight,
    pytest.mark.xdist_group(name="resinsight"),
]


@pytest.fixture(scope="module")
def make_surface():
    """Factory for XTGeo regular surfaces with overridable defaults."""

    def _make(ncol=5, nrow=4, **overrides) -> xtgeo.RegularSurface:
        params = {"xinc": 10.0, "yinc": 20.0, "xori": 100.0, "yori": 200.0}
        params.update(overrides)
        params.setdefault(
            "values", np.arange(ncol * nrow, dtype=np.float64).reshape((ncol, nrow))
        )
        return xtgeo.RegularSurface(ncol=ncol, nrow=nrow, **params)

    return _make


@pytest.fixture(scope="module")
def seeded_surfaces(resinsight_instance, make_surface):
    """Seed 'EXAMPLE_SURF' in the root folder and in a sub-folder.

    Names are unique per folder in ResInsight, so a duplicate name can only
    exist in another folder.
    """
    root = make_surface(xinc=10.0, yinc=10.0, values=np.full((5, 4), 100.0))
    root.to_resinsight(resinsight_instance, surface_name="EXAMPLE_SURF", replace=True)

    sub = make_surface(
        ncol=10, nrow=8, xinc=25.0, yinc=25.0, values=np.full((10, 8), 200.0)
    )
    sub.to_resinsight(
        resinsight_instance,
        surface_name="EXAMPLE_SURF",
        folder_name="EXAMPLE_FOLDER",
        replace=True,
    )


# ---------------------------------------------------------------------------
# regular_surface_from_resinsight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "folder_name, expected_dims",
    [("", (5, 4)), ("EXAMPLE_FOLDER", (10, 8))],
    ids=["root", "sub-folder"],
)
def test_from_resinsight_selects_surface_per_folder(
    resinsight_instance, seeded_surfaces, folder_name, expected_dims
):
    """The folder decides which of the same-named surfaces is loaded."""
    surf = xtgeo.regular_surface_from_resinsight(
        resinsight_instance, "EXAMPLE_SURF", folder_name=folder_name
    )
    assert isinstance(surf, xtgeo.RegularSurface)
    assert (surf.ncol, surf.nrow) == expected_dims


def test_from_resinsight_unknown_name_or_folder_raises(resinsight_instance):
    """Unknown surface and folder names are rejected."""
    with pytest.raises(RuntimeError, match="Cannot find any surface with name"):
        xtgeo.regular_surface_from_resinsight(resinsight_instance, "NO_SUCH_NAME")

    with pytest.raises(RuntimeError, match="Cannot find surface folder"):
        xtgeo.regular_surface_from_resinsight(
            resinsight_instance, "EXAMPLE_SURF", folder_name="NO/SUCH/FOLDER"
        )


def test_from_resinsight_auto_discover(resinsight_instance, seeded_surfaces):
    """Passing None lets ResInsight auto-discover the running instance."""
    discovered = rips.Instance.find()
    if discovered is None or discovered.location != resinsight_instance.location:
        pytest.skip("Auto-discovery found another ResInsight instance on this host")

    assert isinstance(
        xtgeo.regular_surface_from_resinsight(None, "EXAMPLE_SURF"),
        xtgeo.RegularSurface,
    )


def test_from_resinsight_with_property_name(resinsight_instance, make_surface):
    """An explicit property_name loads that property instead of the depth."""
    depth = make_surface()
    depth.to_resinsight(resinsight_instance, surface_name="PROP_API_TEST")

    poro = make_surface(values=np.full((5, 4), 0.3))
    poro.to_resinsight(
        resinsight_instance,
        surface_name="PROP_API_TEST",
        property_name="Porosity",
        set_as_depth=False,
    )

    loaded = xtgeo.regular_surface_from_resinsight(
        resinsight_instance, "PROP_API_TEST", property_name="Porosity"
    )
    assert_allclose(loaded.values, poro.values, atol=1e-4)


# ---------------------------------------------------------------------------
# RegularSurface.to_resinsight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("folder_name", ["", "OUTER/INNER"], ids=["root", "nested"])
def test_to_resinsight_roundtrip(resinsight_instance, make_surface, folder_name):
    """A surface written to ResInsight is read back unchanged."""
    original = make_surface(
        ncol=8, nrow=6, rotation=30.0, values=np.full((8, 6), 1500.0)
    )
    original.to_resinsight(
        resinsight_instance, surface_name="ROUNDTRIP_SURF", folder_name=folder_name
    )

    reloaded = xtgeo.regular_surface_from_resinsight(
        resinsight_instance, "ROUNDTRIP_SURF", folder_name=folder_name
    )

    assert (reloaded.ncol, reloaded.nrow) == (original.ncol, original.nrow)
    assert reloaded.xori == pytest.approx(original.xori)
    assert reloaded.yori == pytest.approx(original.yori)
    assert reloaded.xinc == pytest.approx(original.xinc)
    assert reloaded.rotation == pytest.approx(original.rotation)
    assert_allclose(reloaded.values, original.values, atol=1e-1)


def test_to_resinsight_replaces_only_with_replace_flag(
    resinsight_instance, make_surface
):
    """Replacing a surface with a differing geometry requires replace=True."""
    make_surface(ncol=3, nrow=3).to_resinsight(
        resinsight_instance, surface_name="REPLACE_SURF"
    )
    other = make_surface(ncol=7, nrow=6)

    with pytest.raises(RuntimeError, match="Pass replace=True"):
        other.to_resinsight(resinsight_instance, surface_name="REPLACE_SURF")

    other.to_resinsight(resinsight_instance, surface_name="REPLACE_SURF", replace=True)

    reloaded = xtgeo.regular_surface_from_resinsight(
        resinsight_instance, "REPLACE_SURF"
    )
    assert (reloaded.ncol, reloaded.nrow) == (7, 6)


def test_to_resinsight_normalizes_right_handed_surface(
    resinsight_instance, make_surface
):
    """A yflip=-1 surface keeps its physical geometry through the roundtrip.

    ResInsight has no yflip concept, so the surface is flipped to a left-handed
    layout before export, as done for the RMS interface. That reverses the J
    index direction while the node coordinates and values stay put.
    """
    original = make_surface(ncol=3, nrow=4)
    original.make_righthanded()
    original.to_resinsight(resinsight_instance, surface_name="YFLIP_SURF", replace=True)

    reloaded = xtgeo.regular_surface_from_resinsight(resinsight_instance, "YFLIP_SURF")

    assert reloaded.yinc > 0
    for icol in range(1, original.ncol + 1):
        for jrow in range(1, original.nrow + 1):
            assert_allclose(
                reloaded.get_xy_value_from_ij(icol, original.nrow + 1 - jrow),
                original.get_xy_value_from_ij(icol, jrow),
                atol=1e-2,
            )
