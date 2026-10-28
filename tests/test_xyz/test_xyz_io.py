from io import BytesIO, StringIO

import pytest

from xtgeo.io._file import FileWrapper
from xtgeo.xyz import Points, Polygons
from xtgeo.xyz._xyz_io import export_rms_attr, to_file


@pytest.fixture
def points_with_attrs() -> Points:
    """Create points with each attribute type supported by the exporter.

    The types are given in the ``transl`` mapping in ``export_rms_attr``.
    """
    return Points(
        [
            (1.0, 2.0, 3.0, 1, 0.25, "A"),
            (4.0, 5.0, 6.0, 2, 0.5, "B"),
        ],
        attributes={"code": "int", "value": "float", "region": "str"},
    )


class TestExportRmsAttr:
    """Test RMS attribute export to files and streams, including validation."""

    def test_that_all_attributes_are_exported(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Export all attributes with typed headers and return the row count."""
        output = tmp_path / "all.rmsattr"

        nrows = export_rms_attr(points_with_attrs, FileWrapper(output, mode="w"))

        assert nrows == 2
        assert output.read_text().splitlines() == [
            "Discrete code",
            "Float value",
            "String region",
            "1.0 2.0 3.0 1 0.25 A",
            "4.0 5.0 6.0 2 0.5 B",
        ]

    def test_that_only_selected_attributes_are_exported(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Export only the selected attribute alongside the XYZ coordinates."""
        output = tmp_path / "selected.rmsattr"

        nrows = export_rms_attr(
            points_with_attrs, FileWrapper(output, mode="w"), attributes=["value"]
        )

        assert nrows == 2
        assert output.read_text().splitlines() == [
            "Float value",
            "1.0 2.0 3.0 0.25",
            "4.0 5.0 6.0 0.5",
        ]

    @pytest.mark.parametrize("attributes", [False, None])
    def test_that_only_xyz_is_exported_when_attributes_are_disabled(
        self, points_with_attrs: Points, tmp_path, attributes
    ) -> None:
        """Export XYZ without headers or attributes for False and None."""
        output = tmp_path / "xyz.rmsattr"

        nrows = export_rms_attr(
            points_with_attrs, FileWrapper(output, mode="w"), attributes=attributes
        )

        assert nrows == 2
        assert output.read_text().splitlines() == [
            "1.0 2.0 3.0",
            "4.0 5.0 6.0",
        ]

    def test_that_only_matching_points_are_exported(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Export only matching rows and return the filtered row count."""
        output = tmp_path / "filtered.rmsattr"

        nrows = export_rms_attr(
            points_with_attrs,
            FileWrapper(output, mode="w"),
            pfilter={"region": ["B"]},
        )

        assert nrows == 1
        assert output.read_text().splitlines() == [
            "Discrete code",
            "Float value",
            "String region",
            "4.0 5.0 6.0 2 0.5 B",
        ]

    def test_that_unknown_attributes_raise_value_error(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Raise ValueError when a selected attribute does not exist."""
        with pytest.raises(ValueError, match="not a valid attribute"):
            export_rms_attr(
                points_with_attrs,
                FileWrapper(tmp_path / "invalid.rmsattr", mode="w"),
                attributes=["missing"],
            )

    def test_that_a_bare_attribute_name_raises_type_error(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Reject a bare attribute-name string with TypeError."""
        with pytest.raises(TypeError, match="Attributes must be a bool or a list"):
            export_rms_attr(
                points_with_attrs,
                FileWrapper(tmp_path / "invalid.rmsattr", mode="w"),
                attributes="value",  # type: ignore[arg-type]
            )

    def test_that_unknown_filter_columns_raise_key_error(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Raise KeyError when a filter references a missing column."""
        with pytest.raises(KeyError, match="requested pfilter key missing"):
            export_rms_attr(
                points_with_attrs,
                FileWrapper(tmp_path / "invalid.rmsattr", mode="w"),
                pfilter={"missing": ["x"]},
            )

    def test_that_empty_points_return_zero_without_creating_a_file(
        self, tmp_path
    ) -> None:
        """Return zero and leave the output file absent for empty input."""
        output = tmp_path / "empty.rmsattr"

        assert export_rms_attr(Points(), FileWrapper(output, mode="w")) == 0
        assert not output.exists()

    def test_that_polygon_ids_are_replaced_with_separators(self, tmp_path) -> None:
        """Replace polygon IDs with 999 separators and include them in the count."""
        polygons = Polygons(
            [
                (0.0, 0.0, 1.0, 1),
                (1.0, 0.0, 1.0, 1),
                (2.0, 0.0, 1.0, 2),
                (3.0, 0.0, 1.0, 2),
            ]
        )
        output = tmp_path / "polygons.rmsattr"

        nrows = export_rms_attr(
            polygons, FileWrapper(output, mode="w"), attributes=False
        )

        assert nrows == 6
        assert output.read_text().splitlines() == [
            "0.0 0.0 1.0",
            "1.0 0.0 1.0",
            "999.0 999.0 999.0",
            "2.0 0.0 1.0",
            "3.0 0.0 1.0",
            "999.0 999.0 999.0",
        ]

    @pytest.mark.parametrize("stream_type", [StringIO, BytesIO])
    @pytest.mark.parametrize("attributes", [True, ["value"], False, None])
    def test_that_memory_streams_receive_headers_and_data_without_being_closed(
        self, points_with_attrs: Points, stream_type, attributes
    ) -> None:
        """Write the selected columns to either stream type and keep it open."""
        stream = stream_type()

        nrows = export_rms_attr(
            points_with_attrs, FileWrapper(stream, mode="w"), attributes=attributes
        )

        assert nrows == 2
        assert not stream.closed
        contents = stream.getvalue()
        if isinstance(contents, bytes):
            contents = contents.decode("utf-8")
        if attributes is True:
            expected = [
                "Discrete code",
                "Float value",
                "String region",
                "1.0 2.0 3.0 1 0.25 A",
                "4.0 5.0 6.0 2 0.5 B",
            ]
        elif attributes:
            expected = [
                "Float value",
                "1.0 2.0 3.0 0.25",
                "4.0 5.0 6.0 0.5",
            ]
        else:
            expected = ["1.0 2.0 3.0", "4.0 5.0 6.0"]
        assert contents.splitlines() == expected

    def test_that_existing_file_contents_are_replaced(
        self, points_with_attrs: Points, tmp_path
    ) -> None:
        """Overwrite an existing file rather than append a second export."""
        output = tmp_path / "existing.rmsattr"
        output.write_text("Old contents that must not remain\n" * 10)

        export_rms_attr(
            points_with_attrs, FileWrapper(output, mode="w"), attributes=["value"]
        )

        assert output.read_text().splitlines() == [
            "Float value",
            "1.0 2.0 3.0 0.25",
            "4.0 5.0 6.0 0.5",
        ]


class TestToFile:
    """Test dispatch of XYZ and RMS attribute exports to memory streams."""

    @pytest.mark.parametrize("stream_type", [StringIO, BytesIO])
    @pytest.mark.parametrize("fformat", ["xyz", "rms_attr"])
    def test_that_xyz_and_rms_attr_exports_accept_memory_streams(
        self, points_with_attrs: Points, stream_type, fformat: str
    ) -> None:
        """Pass the wrapper through the dispatcher for both export formats."""
        stream = stream_type()

        nrows = to_file(points_with_attrs, stream, fformat=fformat, attributes=True)

        assert nrows == 2
        assert not stream.closed
        contents = stream.getvalue()
        if isinstance(contents, bytes):
            contents = contents.decode("utf-8")
        if fformat == "xyz":
            expected = ["1.0 2.0 3.0", "4.0 5.0 6.0"]
        else:
            expected = [
                "Discrete code",
                "Float value",
                "String region",
                "1.0 2.0 3.0 1 0.25 A",
                "4.0 5.0 6.0 2 0.5 B",
            ]
        assert contents.splitlines() == expected
