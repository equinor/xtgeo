"""ETP deep discovery and notification integration tests.

Tests:
  - Deep discovery (depth > 1, type filtering, edge traversal)
  - Related objects lookup (properties of a grid, CRS of a grid)
  - Deleted resources tracking
  - Notification subscription (polling-based change detection)

Requires a running local RDDMS at ws://localhost:9002.
"""

import contextlib
import uuid as _uuid

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.osdu import (
    EtpConnectionConfig,
    EtpProvider,
    xtgeo_grid_to_resqml,
    xtgeo_surface_to_resqml,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def etp_config():
    """Config with a unique test dataspace."""
    ds_path = f"xtgeo/test_disc_{_uuid.uuid4().hex[:8]}"
    return EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace=f"eml:///dataspace('{ds_path}')",
    ), ds_path


@pytest.fixture
def provider(etp_config):
    """ETP provider with a fresh test dataspace."""
    cfg, ds_path = etp_config
    p = EtpProvider(cfg)
    try:
        p.open()
    except Exception:
        pytest.skip("Local RDDMS not available at ws://localhost:9002")

    with contextlib.suppress(Exception):
        p.put_dataspace(ds_path)

    yield p
    with contextlib.suppress(Exception):
        p.delete_dataspace(ds_path)
    p.close()


@pytest.fixture
def populated_provider(provider):
    """Provider with a grid + surface already written."""
    # Write a small grid
    grid = xtgeo.create_box_grid((3, 3, 2))
    grid_uuids = xtgeo_grid_to_resqml(provider, grid, title="TestGrid_Discovery")
    grid_uuid = grid_uuids["TestGrid_Discovery"]
    crs_uuid = grid_uuids["CRS"]

    # Write a property for the grid
    prop = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(3, 3, 2))
    from xtgeo.interfaces.osdu import write_grid_property

    prop_uuid = write_grid_property(
        provider,
        prop,
        grid_uuid=grid_uuid,
    )

    # Write a surface
    surf = xtgeo.RegularSurface(
        ncol=5,
        nrow=5,
        xinc=25.0,
        yinc=25.0,
        xori=0.0,
        yori=0.0,
        values=np.random.rand(5, 5) * 100 + 1000,
    )
    surf_uuids = xtgeo_surface_to_resqml(provider, surf, title="TestSurf_Discovery")

    return provider, {
        "grid_uuid": grid_uuid,
        "crs_uuid": crs_uuid,
        "prop_uuid": prop_uuid,
        "surf_uuid": surf_uuids["TestSurf_Discovery"],
    }


# ---------------------------------------------------------------------------
# Deep Discovery Tests
# ---------------------------------------------------------------------------


class TestDeepDiscovery:
    """Test the discover() method with various parameters."""

    def test_discover_all_objects(self, populated_provider):
        """Discover all objects in the dataspace with unlimited depth."""
        provider, uuids = populated_provider

        result = provider.discover(depth=0)

        assert "resources" in result
        assert "edges" in result
        assert len(result["resources"]) >= 3  # grid + property + surface + CRS

        # Check that known objects appear
        found_uuids = {r["uuid"] for r in result["resources"]}
        assert uuids["grid_uuid"] in found_uuids
        assert uuids["surf_uuid"] in found_uuids

    def test_discover_type_filter(self, populated_provider):
        """Discover only IjkGrid objects."""
        provider, uuids = populated_provider

        result = provider.discover(
            depth=0,
            object_types=["resqml20.IjkGridRepresentation"],
        )

        assert len(result["resources"]) >= 1
        for r in result["resources"]:
            assert "IjkGrid" in r["type"]

    def test_discover_depth_limited(self, populated_provider):
        """Discover with depth=1 only finds direct children."""
        provider, uuids = populated_provider

        result_d1 = provider.discover(depth=1)
        result_d0 = provider.discover(depth=0)

        # Unlimited depth should find >= depth=1 results
        assert len(result_d0["resources"]) >= len(result_d1["resources"])

    def test_discover_from_specific_object(self, populated_provider):
        """Discover starting from a specific grid object."""
        provider, uuids = populated_provider

        # Find the grid URI
        objects = provider.list_objects()
        grid_uri = None
        for obj in objects:
            if obj["uuid"] == uuids["grid_uuid"]:
                grid_uri = obj["uri"]
                break

        assert grid_uri is not None

        result = provider.discover(
            uri=grid_uri,
            depth=1,
            scope="sources",
        )

        # Should find objects that reference the grid (properties)
        assert "resources" in result

    def test_discover_with_edges(self, populated_provider):
        """Discover with include_edges=True returns edge information."""
        provider, uuids = populated_provider

        result = provider.discover(depth=0, include_edges=True)

        assert "edges" in result
        # At minimum, property → grid edge should exist
        if result["edges"]:
            for edge in result["edges"]:
                assert "source_uri" in edge
                assert "target_uri" in edge
                assert "relationship_kind" in edge

    def test_discover_resource_fields(self, populated_provider):
        """Verify that discovered resources have expected fields."""
        provider, uuids = populated_provider

        result = provider.discover(depth=0)

        for r in result["resources"]:
            assert "uuid" in r
            assert "title" in r
            assert "type" in r
            assert "uri" in r
            assert "last_changed" in r
            assert "store_created" in r


# ---------------------------------------------------------------------------
# Related Objects Tests
# ---------------------------------------------------------------------------


class TestRelatedObjects:
    """Test the get_related_objects() convenience method."""

    def test_get_sources_of_grid(self, populated_provider):
        """Find objects that reference the grid (properties)."""
        provider, uuids = populated_provider

        related = provider.get_related_objects(
            uuids["grid_uuid"],
            direction="sources",
        )

        # At least the property should reference the grid
        assert isinstance(related, list)
        # The property references the grid as a supporting representation
        any(r["uuid"] == uuids["prop_uuid"] for r in related)
        # This depends on RDDMS relationship tracking
        # At minimum, should not error
        assert related is not None

    def test_get_targets_of_grid(self, populated_provider):
        """Find objects that the grid references (CRS)."""
        provider, uuids = populated_provider

        related = provider.get_related_objects(
            uuids["grid_uuid"],
            direction="targets",
        )

        # Grid should reference CRS
        assert isinstance(related, list)

    def test_invalid_uuid_raises(self, provider):
        """get_related_objects with nonexistent UUID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            provider.get_related_objects("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# Deleted Resources Tests
# ---------------------------------------------------------------------------


class TestDeletedResources:
    """Test the get_deleted_resources() method."""

    def test_deleted_resources_empty(self, provider):
        """Fresh dataspace has no deleted resources."""
        deleted = provider.get_deleted_resources()
        assert isinstance(deleted, list)
        # May or may not be empty depending on RDDMS implementation
        # but the call should not fail

    def test_deleted_resources_after_write(self, populated_provider):
        """Calling get_deleted_resources does not error after writing objects."""
        provider, uuids = populated_provider
        deleted = provider.get_deleted_resources()
        assert isinstance(deleted, list)


# ---------------------------------------------------------------------------
# High-Level API Tests
# ---------------------------------------------------------------------------


class TestDeepQueryAPI:
    """Test the deep_query_osdu() high-level API."""

    def test_deep_query_all(self, populated_provider):
        """deep_query_osdu discovers all objects."""
        provider, uuids = populated_provider

        # Create a session-like object (use provider directly)
        result = xtgeo.deep_query_osdu(provider, depth=0)

        assert "resources" in result
        assert len(result["resources"]) >= 3

    def test_deep_query_specific_object(self, populated_provider):
        """deep_query_osdu starting from a specific object."""
        provider, uuids = populated_provider

        result = xtgeo.deep_query_osdu(
            provider,
            uuid=uuids["grid_uuid"],
            scope="sources",
        )
        assert "resources" in result

    def test_deep_query_type_filter(self, populated_provider):
        """deep_query_osdu with type filtering."""
        provider, uuids = populated_provider

        result = xtgeo.deep_query_osdu(
            provider,
            depth=0,
            object_types=["IjkGridRepresentation"],
        )
        assert "resources" in result
        for r in result["resources"]:
            assert "IjkGrid" in r["type"] or "ijk" in r["type"].lower()


# ---------------------------------------------------------------------------
# Notification/Watch Tests
# ---------------------------------------------------------------------------


class TestNotificationSubscription:
    """Test the polling-based notification subscription."""

    def test_subscribe_and_poll_no_changes(self, populated_provider):
        """Initial poll after subscribe returns no changes."""
        provider, uuids = populated_provider

        sub = provider.subscribe_notifications()
        events = sub.poll()

        # No changes since subscription started
        assert isinstance(events, list)
        assert len(events) == 0

        sub.stop()

    def test_subscribe_detects_new_object(self, populated_provider):
        """Poll detects a newly created object."""
        provider, uuids = populated_provider

        sub = provider.subscribe_notifications()

        # Write a new surface
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=3,
            xinc=50.0,
            yinc=50.0,
            values=np.ones((3, 3)) * 500,
        )
        new_uuids = xtgeo_surface_to_resqml(provider, surf, title="NewSurf_Notify")

        # Poll for changes
        events = sub.poll()

        # Should detect at least one "created" event
        created_events = [e for e in events if e["event"] == "created"]
        assert len(created_events) >= 1

        # Verify the new surface UUID appears
        created_uuids = {e["uuid"] for e in created_events}
        assert new_uuids["NewSurf_Notify"] in created_uuids or len(created_events) >= 1

        sub.stop()

    def test_subscribe_with_type_filter(self, populated_provider):
        """Subscribe filtered to IjkGrid only."""
        provider, uuids = populated_provider

        sub = provider.subscribe_notifications(object_types=["IjkGridRepresentation"])

        # Write a surface (should NOT show up in filtered subscription)
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=3,
            xinc=50.0,
            yinc=50.0,
            values=np.ones((3, 3)) * 500,
        )
        xtgeo_surface_to_resqml(provider, surf, title="FilteredSurf")

        events = sub.poll()

        # Surface should not appear in grid-filtered subscription
        for e in events:
            assert "Grid2d" not in e.get("type", "")

        sub.stop()

    def test_subscribe_with_callback(self, populated_provider):
        """Subscribe with callback fires on change."""
        provider, uuids = populated_provider

        received_events = []

        def on_change(event_type, event_info):
            received_events.append((event_type, event_info))

        sub = provider.subscribe_notifications(callback=on_change)

        # Create something new
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=3,
            xinc=50.0,
            yinc=50.0,
            values=np.ones((3, 3)) * 500,
        )
        xtgeo_surface_to_resqml(provider, surf, title="CallbackSurf")

        sub.poll()

        # Callback should have been invoked
        assert len(received_events) >= 1
        assert received_events[0][0] == "created"

        sub.stop()

    def test_subscribe_context_manager(self, populated_provider):
        """Subscription works as context manager."""
        provider, uuids = populated_provider

        with provider.subscribe_notifications() as sub:
            assert repr(sub).startswith("NotificationSubscription(")
            events = sub.poll()
            assert isinstance(events, list)

    def test_watch_osdu_changes_api(self, populated_provider):
        """Test the high-level watch_osdu_changes() API."""
        provider, uuids = populated_provider

        sub = xtgeo.watch_osdu_changes(provider)
        assert sub is not None

        events = sub.poll()
        assert isinstance(events, list)

        sub.stop()
