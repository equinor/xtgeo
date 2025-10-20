"""Tests for RegularSurfaceReader and RegularSurfaceWriter classes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

# Skip if RMS API not available
pytest.importorskip("rmsapi", reason="RMS API not available")

from xtgeo.common.log import null_logger
from xtgeo.interfaces.rms._regular_surface import (
    RegularSurfaceDataRms,
    RegularSurfaceReader,
    RegularSurfaceWriter,
)
from xtgeo.interfaces.rms._rmsapi_package import rmsapi
from xtgeo.interfaces.rms.rmsapi_utils import _StorageTypeRegularSurface as StorageType

logger = null_logger(__name__)


@pytest.mark.requires_roxar
class TestRegularSurfaceReaderInit:
    """Test RegularSurfaceReader initialization and basic functionality."""

    @staticmethod
    def test_reader_with_horizons_surface(rms_regular_surface_project):
        """Test loading surface from horizons."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )

        data = reader.load()

        assert isinstance(data, RegularSurfaceDataRms)

        # Check that request parameters were kept
        assert reader.name == "TOP1"
        assert reader.category == "CategoryDepth"
        assert reader.stype == "horizons"
        assert reader.realisation == 0

        assert reader.readonly is True

        # Check geometry was loaded
        assert data.xori == pytest.approx(0.0)
        assert data.yori == pytest.approx(0.0)
        assert data.ncol == 11
        assert data.nrow == 11
        assert data.xinc == pytest.approx(10.0)
        assert data.yinc == pytest.approx(10.0)
        assert data.rotation == pytest.approx(15.0)

        # Check values were loaded
        assert data.values is not None
        assert isinstance(data.values, np.ma.MaskedArray)
        assert data.values.shape == (11, 11)

    @staticmethod
    def test_reader_with_zones_surface(rms_regular_surface_project):
        """Test loading surface from zones."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="ISO_TOP",
            category="CategoryZoneDepth",
            stype="zones",
            realisation=0,
        )

        data = reader.load()

        assert reader.name == "ISO_TOP"
        assert reader.category == "CategoryZoneDepth"
        assert reader.stype == "zones"
        assert data.ncol == 11
        assert data.nrow == 11

    @staticmethod
    def test_reader_with_clipboard_surface(rms_regular_surface_project):
        """Test loading surface from clipboard."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1",
            category="",  # Clipboard doesn't require category
            stype="clipboard",
            realisation=0,
        )

        data = reader.load()

        assert reader.name == "Surface1"
        assert reader.category == ""
        assert reader.stype == "clipboard"
        assert data.ncol == 11
        assert data.nrow == 11

    @staticmethod
    def test_reader_with_general2d_data_surface(rms_regular_surface_project):
        """Test loading surface from general2d_data."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1",
            category="",
            stype="general2d_data",
            realisation=0,
        )

        data = reader.load()

        assert reader.name == "Surface1"
        assert reader.stype == "general2d_data"
        assert data.ncol == 11
        assert data.nrow == 11

    @staticmethod
    def test_reader_with_project_path(rms_project_as_folder_path):
        """Test reader with project path instead of project object."""
        # This should fail because the basic project doesn't have surfaces
        reader = RegularSurfaceReader(
            project_or_path=rms_project_as_folder_path,
            name="NonExistentSurface",
            category="NonExistentCategory",
            stype="horizons",
            realisation=0,
        )
        with pytest.raises(RuntimeError, match="Failed to load surface"):
            _ = reader.load()


@pytest.mark.requires_roxar
class TestRegularSurfaceReaderValidation:
    """Test input validation in RegularSurfaceReader."""

    @staticmethod
    def test_invalid_stype(rms_regular_surface_project):
        """Test with invalid storage type."""
        with pytest.raises(
            ValueError, match="Given stype 'invalid_type' is not supported"
        ):
            _ = RegularSurfaceReader(
                project_or_path=rms_regular_surface_project,
                name="TOP1",
                category="CategoryDepth",
                stype="invalid_type",
                realisation=0,
            )

    @staticmethod
    def test_empty_name(rms_regular_surface_project):
        """Test with empty surface name."""
        with pytest.raises(ValueError, match="The name is missing or empty"):
            _ = RegularSurfaceReader(
                project_or_path=rms_regular_surface_project,
                name="",
                category="CategoryDepth",
                stype="horizons",
                realisation=0,
            )

    @staticmethod
    def test_missing_category_for_horizons(rms_regular_surface_project):
        """Test that horizons require category."""
        with pytest.raises(
            ValueError, match="Need to specify both name and category for horizons"
        ):
            _ = RegularSurfaceReader(
                project_or_path=rms_regular_surface_project,
                name="TOP1",
                category="",
                stype="horizons",
                realisation=0,
            )

    @staticmethod
    def test_missing_category_for_zones(rms_regular_surface_project):
        """Test that zones require category."""
        with pytest.raises(
            ValueError, match="Need to specify both name and category for horizons"
        ):
            _ = RegularSurfaceReader(
                project_or_path=rms_regular_surface_project,
                name="ISO_TOP",
                category="",
                stype="zones",
                realisation=0,
            )

    @staticmethod
    def test_nonexistent_surface_name(rms_regular_surface_project):
        """Test with non-existent surface name."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="NonExistentSurface",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        with pytest.raises(RuntimeError, match="Failed to load surface"):
            _ = reader.load()

    @staticmethod
    def test_nonexistent_category(rms_regular_surface_project):
        """Test with non-existent category."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="NonExistentCategory",
            stype="horizons",
            realisation=0,
        )
        with pytest.raises(RuntimeError, match="Failed to load surface"):
            _ = reader.load()

    @staticmethod
    def test_invalid_realisation(rms_regular_surface_project):
        """Test with invalid realisation number."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=999,  # Non-existent realisation
        )
        with pytest.raises(RuntimeError, match="Failed to load surface"):
            _ = reader.load()


@pytest.mark.requires_roxar
class TestRegularSurfaceReaderDataValidation:
    """Test that loaded data is correct."""

    @staticmethod
    def test_surface_geometry_values(rms_regular_surface_project):
        """Test that surface geometry values are correct."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )

        data = reader.load()

        # Values should match what was set in the fixture
        assert data.xori == pytest.approx(0.0)
        assert data.yori == pytest.approx(0.0)
        assert data.ncol == 11
        assert data.nrow == 11
        assert data.xinc == pytest.approx(10.0)
        assert data.yinc == pytest.approx(10.0)
        assert data.rotation == pytest.approx(15.0)

    @staticmethod
    def test_surface_values_data(rms_regular_surface_project):
        """Test that surface values data is correct."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )

        data = reader.load()

        assert data.values is not None
        assert isinstance(data.values, np.ma.MaskedArray)
        assert data.values.shape == (11, 11)
        assert data.values.dtype == np.float64

        # Check that values are reasonable (should be sequential from fixture)
        assert np.all(np.isfinite(np.asanyarray(data.values)))

    @staticmethod
    def test_different_surfaces_have_different_values(rms_regular_surface_project):
        """Test that different surfaces have different values."""
        reader1 = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        data1 = reader1.load()

        reader2 = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP2",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        data2 = reader2.load()

        # H2 values should be H1 values + 100 (from fixture)
        assert not np.array_equal(data1.values, data2.values)
        assert np.allclose(data2.values, data1.values + 100.0)


@pytest.mark.requires_roxar
class TestRegularSurfaceReaderMethods:
    """Test RegularSurfaceReader methods."""

    @staticmethod
    def test_cleanup_after_loading(rms_regular_surface_project):
        """Test that cleanup happens after loading."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )

        _ = reader.load()

        # After successful loading, internal utils should be cleaned up
        assert reader._rmsapi_utils is None


@pytest.mark.requires_roxar
class TestRegularSurfaceReaderEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_reader_with_none_project():
        """Test reader with None project."""
        with pytest.raises(RuntimeError, match="Project is not valid"):
            _ = RegularSurfaceReader(
                project_or_path=None,
                name="TestSurface",
                category="TestCategory",
                stype="horizons",
            )

    @staticmethod
    def test_case_insensitive_stype(rms_regular_surface_project):
        """Test that stype is case insensitive."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="HORIZONS",  # Uppercase
            realisation=0,
        )
        data = reader.load()

        assert reader.stype == "HORIZONS"  # Original case preserved
        assert reader._stype_enum == StorageType.HORIZONS
        assert reader._stype_enum.value == "horizons"
        assert data.ncol == 11  # And loading still works

    @staticmethod
    def test_reader_dataclass_fields(rms_regular_surface_project):
        """Test that reader returns dataclass with expected types."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )

        data = reader.load()

        # Check all fields exist and have expected types
        assert isinstance(reader._project, object)  # RMS project object or path
        assert isinstance(reader.name, str)
        assert isinstance(reader.category, str)
        assert isinstance(reader.stype, str)
        assert isinstance(reader.realisation, int)

        assert isinstance(data.name, str)
        assert isinstance(data.xori, float)
        assert isinstance(data.yori, float)
        assert isinstance(data.ncol, int)
        assert isinstance(data.nrow, int)
        assert isinstance(data.xinc, float)
        assert isinstance(data.yinc, float)
        assert isinstance(data.rotation, float)
        assert isinstance(data.values, np.ma.MaskedArray)

    @staticmethod
    def test_reader_with_points_geometry_raises(rms_regular_surface_project):
        """Test that trying to read a points geometry as surface raises error."""
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP2",
            category="CategoryPoints",  # This representation is points, not surface
            stype="horizons",
            realisation=0,
        )
        with pytest.raises(RuntimeError, match="but got Points"):
            _ = reader.load()


# ======================================================================================
# Tests for RegularSurfaceWriter
# ======================================================================================


@pytest.mark.requires_roxar
class TestRegularSurfaceWriter:
    """Tests for writing regular surfaces to RMS via RegularSurfaceWriter."""

    @staticmethod
    def test_write_clipboard_roundtrip(rms_regular_surface_project):
        """Write a clipboard surface and read it back."""
        name = "T_Write_Clipboard"
        category = ""  # clipboard root
        stype = "clipboard"

        # Geometry
        xori, yori = 1000.0, 2000.0
        ncol, nrow = 5, 4
        xinc, yinc = 25.0, 30.0
        rotation = 5.0

        # Values with mask + NaN/Inf to test sanitation
        vals = np.arange(ncol * nrow, dtype=np.float64).reshape(ncol, nrow)
        vals[0, 0] = np.nan
        vals[3, 2] = np.inf
        mvals = np.ma.array(vals, mask=False)
        mvals.mask[2, 1] = True  # one masked cell

        data = RegularSurfaceDataRms(
            name=name,
            xori=xori,
            yori=yori,
            ncol=ncol,
            nrow=nrow,
            xinc=xinc,
            yinc=yinc,
            rotation=rotation,
            values=mvals,
        )

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype=stype,
            realisation=0,
        )
        writer.save(data)

        # Read back and validate
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype=stype,
            realisation=0,
        )
        out = reader.load()

        assert reader.readonly is True
        assert writer.readonly is False

        assert out.name == name
        assert out.ncol == ncol and out.nrow == nrow
        assert out.xori == pytest.approx(xori)
        assert out.yori == pytest.approx(yori)
        assert out.xinc == pytest.approx(xinc)
        assert out.yinc == pytest.approx(yinc)
        assert out.rotation == pytest.approx(rotation)

        assert isinstance(out.values, np.ma.MaskedArray)
        assert out.values.shape == (ncol, nrow)

        # All stored values must be finite in the data buffer
        assert np.isfinite(np.asanyarray(out.values)).all()

        # Mask should include the originally masked cell,
        # and the NaN/Inf cells should be masked after sanitation
        assert out.values.mask[2, 1]
        assert out.values.mask[0, 0]
        assert out.values.mask[3, 2]

    @staticmethod
    def test_write_general2d_roundtrip_or_skip(rms_regular_surface_project):
        """Write a general2d_data surface + folders, or skip if API not supported."""
        name = "T_Write_G2D"
        category = "folder/sub"
        stype = "general2d_data"

        xori, yori = 0.0, 0.0
        ncol, nrow = 3, 3
        xinc, yinc = 10.0, 10.0
        rotation = 0.0

        vals = np.arange(ncol * nrow, dtype=np.float64).reshape(ncol, nrow)
        mvals = np.ma.array(vals, mask=False)

        data = RegularSurfaceDataRms(
            name=name,
            xori=xori,
            yori=yori,
            ncol=ncol,
            nrow=nrow,
            xinc=xinc,
            yinc=yinc,
            rotation=rotation,
            values=mvals,
        )

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype=stype,
            realisation=0,
        )

        try:
            writer.save(data)
        except RuntimeError as exc:
            # If API version is too old, writer raises RuntimeError
            # wrapping NotImplementedError
            if "API Support for general2d_data is missing" in str(exc):
                pytest.skip("general2d_data not supported by this RMS API version")
            raise

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype=stype,
            realisation=0,
        )
        out = reader.load()
        assert np.array_equal(np.asanyarray(out.values), np.asanyarray(vals))

    @staticmethod
    def test_write_horizons_missing_category_raises(rms_regular_surface_project):
        """Horizons require category; writer should fail cleanly."""

        with pytest.raises(
            ValueError, match="Need to specify both name and category for horizons"
        ):
            _ = RegularSurfaceWriter(
                project_or_path=rms_regular_surface_project,
                name="AnyName",
                category="",
                stype="horizons",
                realisation=0,
            )

    @staticmethod
    def test_write_invalid_payload_shape_raises(rms_regular_surface_project):
        """Values shape must match (ncol, nrow)."""
        data = RegularSurfaceDataRms(
            name="BadShape",
            xori=0.0,
            yori=0.0,
            ncol=4,
            nrow=3,
            xinc=1.0,
            yinc=1.0,
            rotation=0.0,
            values=np.ma.zeros((3, 3), dtype=np.float64),  # mismatch
        )

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="BadShape",
            category="",
            stype="clipboard",
            realisation=0,
        )

        with pytest.raises(RuntimeError, match="Failed to save surface"):
            writer.save(data)


@pytest.mark.requires_roxar
class TestRegularSurfaceMoreReaderPlusWriter:
    @staticmethod
    def test_read_write_to_same_object(rms_regular_surface_project):
        """Read data, modify, then write back to same object."""

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        data = reader.load()

        # Modify values
        new_values = data.values + 10.0
        updated = replace(data, values=new_values)

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        writer.save(updated)

        # Read back and verify changes
        reader2 = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        out = reader2.load()

        assert np.array_equal(
            np.asanyarray(out.values), np.asanyarray(data.values + 10.0)
        )

    @staticmethod
    def test_read_write_to_same_object_time_domain(rms_regular_surface_project):
        """Read data, modify, then write back to same object.

        Default domain is depth, here we test with a time-domain surface. Here we
        just verify that read-modify-write works as expected.

        """

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryDepth",
            stype="horizons",
            realisation=0,
        )
        data = reader.load()

        # Modify values
        new_values = data.values + 10.0
        updated = replace(data, values=new_values)

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryTime",
            stype="horizons",
            realisation=0,
        )
        writer.save(updated)

        # Read back and verify changes
        reader2 = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="TOP1",
            category="CategoryTime",
            stype="horizons",
            realisation=0,
        )
        out = reader2.load()

        assert np.array_equal(np.asanyarray(updated.values), np.asanyarray(out.values))

    @staticmethod
    def test_read_write_to_same_object_clipboard(rms_regular_surface_project):
        """Read data, modify, then write back to same object on clipboard."""

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1",  # this is stored with VerticalDomain.time
            category="",
            stype="clipboard",
            realisation=0,
        )
        data = reader.load()
        assert data.ncol == 11 and data.nrow == 11

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="Surface1",
            category="",
            stype="clipboard",
            realisation=0,
        )
        writer.save(data)

        # Read back and verify changes
        reader2 = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1",
            category="",
            stype="clipboard",
            realisation=0,
        )
        out = reader2.load()

        assert np.array_equal(np.asanyarray(data.values), np.asanyarray(out.values))

    @staticmethod
    def test_read_write_to_same_object_in_folder_clipboard(rms_regular_surface_project):
        """Read data, modify, then write back to same object on clipboard."""

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",  # this is stored with VerticalDomain.time
            category="Folder1",
            stype="clipboard",
            realisation=0,
        )
        data = reader.load()
        assert data.ncol == 11 and data.nrow == 11

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",
            category="Folder1",
            stype="clipboard",
            realisation=0,
        )
        writer.save(data)

        # Read back and verify changes
        reader2 = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",
            category="Folder1",
            stype="clipboard",
            realisation=0,
        )
        out = reader2.load()

        assert np.array_equal(np.asanyarray(data.values), np.asanyarray(out.values))

    @staticmethod
    def test_change_domain_clipboard(rms_regular_surface_project):
        """Change domain while writing."""

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",  # this is stored with VerticalDomain.time
            category="Folder1",
            stype="clipboard",
            realisation=0,
        )
        data = reader.load()
        assert data.ncol == 11 and data.nrow == 11

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",
            category="Folder1",
            stype="clipboard",
            realisation=0,
            domain="depth",  # change from time to depth
        )
        writer.save(data)

        # need to check the domain by using RMSAPI directly
        proj = rms_regular_surface_project
        item = proj.clipboard.folders["Folder1"]["Surface1_folder"]

        assert item.vertical_domain == rmsapi.VerticalDomain.depth

    @staticmethod
    def test_change_domain_invalid_clipboard(rms_regular_surface_project):
        """Invalid domain type"""

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",  # this is stored with VerticalDomain.time
            category="Folder1",
            stype="clipboard",
            realisation=0,
        )
        data = reader.load()
        assert data.ncol == 11 and data.nrow == 11

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="Surface1_folder",
            category="Folder1",
            stype="clipboard",
            realisation=0,
            domain="invalid_type",
        )

        with pytest.raises(RuntimeError, match="Failed to save surface"):
            writer.save(data)


@pytest.mark.requires_roxar
class TestRegularSurfaceAdditional:
    @staticmethod
    def test_reader_cleanup_after_failure(rms_regular_surface_project):
        """Cleanup should occur even when load fails."""
        with pytest.raises(
            ValueError, match="Given stype 'invalid_type' is not supported"
        ):
            reader = RegularSurfaceReader(
                project_or_path=rms_regular_surface_project,
                name="TOP1",
                category="CategoryDepth",
                stype="invalid_type",  # force failure in validation
                realisation=0,
            )
            # After failure, utils should be cleaned up
            assert reader._rmsapi_utils is None

    @staticmethod
    def test_write_and_read_clipboard_nested_pipe_path(
        rms_regular_surface_project,
    ):
        """Write and then read from clipboard using a nested '|' folder path."""
        # Reuse existing clipboard surface values
        base_reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1",
            category="",
            stype="clipboard",
            realisation=0,
        )
        data = base_reader.load()

        name = "Surface1_pipe_nested"
        category = "FolderP|SubP"

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype="clipboard",
            realisation=0,
            domain="time",
        )
        writer.save(data)

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype="clipboard",
            realisation=0,
        )
        out = reader.load()

        assert out.values.shape == data.values.shape
        assert isinstance(out.values, np.ma.MaskedArray)

    @staticmethod
    def test_write_and_read_general2d_nested_pipe_path_or_skip(
        rms_regular_surface_project,
    ):
        """Write and read a general2d_data surface using a '|' nested path.

        Skip gracefully if the API lacks general2d_data support.
        """
        name = "T_Write_G2D_Pipe"
        category = "folder|sub2"

        ncol, nrow = 3, 2
        vals = np.arange(ncol * nrow, dtype=np.float64).reshape(ncol, nrow)
        mvals = np.ma.array(vals, mask=False)

        data = RegularSurfaceDataRms(
            name=name,
            xori=0.0,
            yori=0.0,
            ncol=ncol,
            nrow=nrow,
            xinc=10.0,
            yinc=10.0,
            rotation=0.0,
            values=mvals,
        )

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype="general2d_data",
            realisation=0,
        )

        try:
            writer.save(data)
        except RuntimeError as exc:
            if "API Support for general2d_data is missing" in str(exc):
                pytest.skip("general2d_data not supported by this RMS API version")
            raise

        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name=name,
            category=category,
            stype="general2d_data",
            realisation=0,
        )
        out = reader.load()
        assert np.array_equal(np.asanyarray(out.values), np.asanyarray(vals))

    @staticmethod
    def test_write_trends_with_category_warns(rms_regular_surface_project):
        """Trends should warn when category is provided."""
        # Read from an existing surface for values
        reader = RegularSurfaceReader(
            project_or_path=rms_regular_surface_project,
            name="Surface1",
            category="",
            stype="clipboard",
        )
        data = reader.load()

        writer = RegularSurfaceWriter(
            project_or_path=rms_regular_surface_project,
            name="TrendSurface",
            category="SomeCategory",  # Should trigger warning
            stype="trends",
        )

        with pytest.warns(UserWarning, match="Ignoring category"):
            writer.save(data)


@pytest.mark.requires_roxar
class TestRegularSurfaceDataValidation:
    """Test RegularSurfaceDataRms validation without full RMS setup."""

    @staticmethod
    def test_validate_payload_negative_ncol():
        """Negative ncol should raise ValueError."""
        data = RegularSurfaceDataRms(
            name="test",
            xori=0.0,
            yori=0.0,
            ncol=-1,
            nrow=2,
            xinc=1.0,
            yinc=1.0,
            rotation=0.0,
            values=np.ma.zeros((2, 2)),
        )
        with pytest.raises(ValueError, match="ncol and nrow must be positive"):
            RegularSurfaceWriter._validate_payload(data)

    @staticmethod
    def test_sanitize_converts_dtype():
        """Sanitizer converts float32 to float64."""
        vals = np.ma.array([[1.0, 2.0]], dtype=np.float32)
        result = RegularSurfaceWriter._sanitize_values_for_rms(vals)
        assert result.dtype == np.float64

    @staticmethod
    def test_sanitize_replaces_nan_inf():
        """NaN and Inf become masked."""
        vals = np.ma.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float64)
        result = RegularSurfaceWriter._sanitize_values_for_rms(vals)
        assert result.mask[0, 1]  # NaN position
        assert result.mask[1, 0]  # Inf position
        assert not result.mask[0, 0]
        assert not result.mask[1, 1]


if __name__ == "__main__":
    pytest.main([__file__])
