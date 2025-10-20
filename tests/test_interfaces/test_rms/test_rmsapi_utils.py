"""Tests for RmsApiUtils class."""

from __future__ import annotations

import warnings

import pytest

from xtgeo.common.log import null_logger
from xtgeo.interfaces.rms.rmsapi_utils import (
    RmsApiUtils,
    RoxUtils,
    _DomainType,
    _StorageTypeRegularSurface,
)

logger = null_logger(__name__)


@pytest.mark.requires_roxar
class TestRmsApiUtilsInit:
    """Test RmsApiUtils initialization."""

    def test_init_with_project_path(self, rms_project_as_folder_path):
        """Test initialization with project path."""
        assert isinstance(rms_project_as_folder_path, str)
        utils = RmsApiUtils(rms_project_as_folder_path, readonly=True)
        assert utils.project is not None
        assert utils.project.name == "Dummy Project as folder path"
        logger.info("RMS API version: %s", utils.rmsapiversion)
        assert utils.rmsapiversion is not None
        assert utils._rmsexternal is True
        utils.safe_close()

    def test_init_with_project_object(self, rms_project_as_project):
        """Test initialization with project object."""
        utils = RmsApiUtils(rms_project_as_project)
        assert utils.project is rms_project_as_project
        assert utils.project.name == "Dummy Project as project"
        assert utils.rmsapiversion is not None
        assert utils._rmsexternal is False
        # Don't need to call safe_close() here as it won't close the project

    def test_init_invalid_project(self):
        """Test initialization with invalid project."""
        with pytest.raises(RuntimeError, match="Project is not valid"):
            RmsApiUtils(123)  # Invalid project type

    def test_init_none_project(self):
        """Test initialization with None project."""
        with pytest.raises(RuntimeError, match="Project is not valid"):
            RmsApiUtils(None)


@pytest.mark.requires_roxar
class TestRmsApiUtilsProperties:
    """Test RmsApiUtils properties."""

    def test_version_properties(self, rms_project_as_folder_path):
        """Test version-related properties."""
        utils = RmsApiUtils(rms_project_as_folder_path, readonly=True)

        # Test version properties
        assert isinstance(utils.rmsapiversion, str)
        assert utils.roxversion == utils.rmsapiversion  # Backward compatibility
        assert "." in utils.rmsapiversion  # Should look like "1.x"

        utils.safe_close()

    def test_project_property(self, rms_project_as_folder_path):
        """Test project property."""
        utils = RmsApiUtils(rms_project_as_folder_path, readonly=True)

        project = utils.project
        assert project is not None
        assert hasattr(project, "horizons")
        assert hasattr(project, "zones")

        utils.safe_close()


@pytest.mark.requires_roxar
class TestRmsApiUtilsVersionMethods:
    """Test version-related methods."""

    def test_version_required(self, rms_project_as_project):
        """Test version_required method."""
        utils = RmsApiUtils(rms_project_as_project, readonly=True)

        # Test with older version (should be True for current versions)
        assert utils.version_required("1.0") is True
        assert utils.version_required("1.5") is True

        # Test with future version (should be False)
        assert utils.version_required("99.0") is False

    def test_rmsversion(self, rms_project_as_project):
        """Test rmsversion method."""
        utils = RmsApiUtils(rms_project_as_project, readonly=True)

        # Test known API versions
        rms_versions = utils.rmsversion("1.5")
        assert rms_versions is not None
        assert isinstance(rms_versions, list)
        assert len(rms_versions) > 0


@pytest.mark.requires_roxar
class TestRmsApiUtilsCategoryManagement:
    """Test category management methods."""

    def test_create_horizons_category(self, rms_project_as_folder_path):
        """Test creating horizons categories."""
        utils = RmsApiUtils(rms_project_as_folder_path)

        # Test single category
        test_cat = "TestHorizonCategory"
        utils.create_horizons_category(test_cat)

        # Verify category exists
        assert test_cat in utils.project.horizons.representations

        # Test multiple categories
        test_cats = ["TestCat1", "TestCat2"]
        utils.create_horizons_category(test_cats)

        for cat in test_cats:
            assert cat in utils.project.horizons.representations

        utils.safe_close()

    def test_create_zones_category(self, rms_project_as_folder_path):
        """Test creating zones categories."""
        utils = RmsApiUtils(rms_project_as_folder_path)

        # Test single category
        test_cat = "TestZoneCategory"
        utils.create_zones_category(test_cat)

        # Verify category exists
        assert test_cat in utils.project.zones.representations

        utils.safe_close()

    def test_create_category_different_domains(self, rms_project_as_folder_path):
        """Test creating categories with different domains."""
        utils = RmsApiUtils(rms_project_as_folder_path)

        # Test different domains
        utils.create_horizons_category("DepthCat", domain="depth")
        utils.create_horizons_category("TimeCat", domain="time")

        assert "DepthCat" in utils.project.horizons.representations
        assert "TimeCat" in utils.project.horizons.representations

        utils.safe_close()

    def test_create_category_different_htypes(self, rms_project_as_folder_path):
        """Test creating categories with different horizon types."""
        utils = RmsApiUtils(rms_project_as_folder_path)

        # Test different horizon types
        utils.create_horizons_category("SurfaceCat", htype="surface")
        utils.create_horizons_category("PointsCat", htype="points")
        utils.create_horizons_category("LinesCat", htype="lines")

        assert "SurfaceCat" in utils.project.horizons.representations
        assert "PointsCat" in utils.project.horizons.representations
        assert "LinesCat" in utils.project.horizons.representations

        utils.safe_close()

    def test_delete_horizons_category(self, rms_project_as_folder_path):
        """Test deleting horizons categories."""
        utils = RmsApiUtils(rms_project_as_folder_path)

        # Create then delete category
        test_cat = "TestDeleteCategory"
        utils.create_horizons_category(test_cat)
        assert test_cat in utils.project.horizons.representations

        utils.delete_horizons_category(test_cat)
        assert test_cat not in utils.project.horizons.representations

        utils.safe_close()

    def test_clear_horizon_category(self, rms_project_as_folder_path):
        """Test clearing horizon categories."""
        utils = RmsApiUtils(rms_project_as_folder_path)

        # Create category and test clearing
        test_cat = "TestClearCategory"
        utils.create_horizons_category(test_cat)

        # This should not raise an error
        utils.clear_horizon_category(test_cat)

        utils.safe_close()


class TestRoxUtilsBackwardCompatibility:
    """Test RoxUtils backward compatibility."""

    @pytest.mark.requires_roxar
    def test_roxutils_deprecation_warning(self, rms_project_as_folder_path):
        """Test that RoxUtils shows deprecation warning."""
        with pytest.warns(PendingDeprecationWarning, match="RoxUtils is deprecated"):
            utils = RoxUtils(rms_project_as_folder_path, readonly=True)
            utils.safe_close()

    @pytest.mark.requires_roxar
    def test_roxutils_functionality(self, rms_project_as_folder_path):
        """Test that RoxUtils still works functionally."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)

            utils = RoxUtils(rms_project_as_folder_path, readonly=True)

            # Should have same functionality as RmsApiUtils
            assert hasattr(utils, "rmsapiversion")
            assert hasattr(utils, "roxversion")
            assert hasattr(utils, "project")
            assert hasattr(utils, "version_required")

            utils.safe_close()


class TestEnums:
    """Test enum classes."""

    def test_domain_type_enum(self):
        """Test DomainType enum."""
        assert _DomainType.DEPTH.value == "depth"
        assert _DomainType.TIME.value == "time"
        assert _DomainType.THICKNESS.value == "thickness"
        assert _DomainType.UNKNOWN.value == "unknown"

        # Test values() method
        values = _DomainType.values()
        assert "depth" in values
        assert "time" in values
        assert "thickness" in values
        assert "unknown" in values

    def test_storage_type_regular_surface_enum(self):
        """Test StorageTypeRegularSurface enum."""
        assert _StorageTypeRegularSurface.HORIZONS.value == "horizons"
        assert _StorageTypeRegularSurface.ZONES.value == "zones"
        assert _StorageTypeRegularSurface.CLIPBOARD.value == "clipboard"
        assert _StorageTypeRegularSurface.GENERAL2D_DATA.value == "general2d_data"
        assert _StorageTypeRegularSurface.TRENDS.value == "trends"

        # Test values() method
        values = _StorageTypeRegularSurface.values()
        assert "horizons" in values
        assert "zones" in values
        assert "clipboard" in values
        assert "general2d_data" in values
        assert "trends" in values


if __name__ == "__main__":
    pytest.main([__file__])
