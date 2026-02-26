import pytest

from xtgeo.common_geometry.bounding_box_3d import BoundingBox3D


def test_bounding_box_properties():
    bbox = BoundingBox3D((0.0, 2.0, 1.0, 4.0, -1.0, 1.0))

    assert bbox.min_x == 0.0
    assert bbox.max_x == 2.0
    assert bbox.min_y == 1.0
    assert bbox.max_y == 4.0
    assert bbox.min_z == -1.0
    assert bbox.max_z == 1.0

    assert bbox.length_x == pytest.approx(2.0)
    assert bbox.length_y == pytest.approx(3.0)
    assert bbox.length_z == pytest.approx(2.0)
    assert bbox.volume == pytest.approx(12.0)
    assert bbox.center == pytest.approx((1.0, 2.5, 0.0))


def test_bounding_box_validation_none():
    with pytest.raises(ValueError, match="values cannot be None"):
        BoundingBox3D(None)
    with pytest.raises(ValueError, match="min_x and max_x cannot be None"):
        BoundingBox3D((None, 1.0, 0.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="min_y and max_y cannot be None"):
        BoundingBox3D((0.0, 1.0, None, 2.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="min_z and max_z cannot be None"):
        BoundingBox3D((0.0, 1.0, 0.0, 1.0, None, 3.0))


def test_bounding_box_validation():
    with pytest.raises(
        ValueError, match="min_x must be smaller than or equal to max_x"
    ):
        BoundingBox3D((1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
    with pytest.raises(
        ValueError, match="min_y must be smaller than or equal to max_y"
    ):
        BoundingBox3D((0.0, 1.0, 2.0, 1.0, 0.0, 1.0))
    with pytest.raises(
        ValueError, match="min_z must be smaller than or equal to max_z"
    ):
        BoundingBox3D((0.0, 1.0, 0.0, 1.0, 3.0, 2.0))


def test_bounding_box_contains_point():
    bbox = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))

    assert bbox.contains_point((0.0, 0.5, 1.0))
    assert bbox.contains_point((1.0, 1.0, 1.0))
    assert not bbox.contains_point((-0.1, 0.5, 0.5))
    assert not bbox.contains_point((0.5, 1.1, 0.5))


def test_bounding_box_intersects_union():
    bbox_a = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    bbox_b = BoundingBox3D((0.5, 2.0, -1.0, 0.5, 0.0, 2.0))
    bbox_c = BoundingBox3D((2.1, 3.0, 2.1, 3.0, 2.1, 3.0))
    bbox_touch = BoundingBox3D((1.0, 2.0, 1.0, 2.0, 1.0, 2.0))

    assert bbox_a.intersects(bbox_b)
    assert not bbox_a.intersects(bbox_c)
    assert bbox_a.intersects(bbox_touch)

    union = bbox_a.union(bbox_b)
    assert union == BoundingBox3D((0.0, 2.0, -1.0, 1.0, 0.0, 2.0))

    union_self = bbox_a.union(bbox_a)
    assert union_self is bbox_a


def test_bounding_box_equality_and_inequality():
    bbox_a = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    bbox_b = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    bbox_c = BoundingBox3D((0.0, 2.0, 0.0, 1.0, 0.0, 1.0))

    assert bbox_a == bbox_a
    assert bbox_a == bbox_b
    assert bbox_a != bbox_c
    assert bbox_a != "not-a-bbox"


def test_bounding_box_intersection():
    bbox_a = BoundingBox3D((0.0, 2.0, 0.0, 2.0, 0.0, 2.0))
    bbox_b = BoundingBox3D((1.0, 3.0, -1.0, 1.0, 0.5, 1.5))
    bbox_c = BoundingBox3D((3.1, 4.0, 3.1, 4.0, 3.1, 4.0))

    intersection = bbox_a.intersection(bbox_b)
    assert intersection is not None
    assert intersection == BoundingBox3D((1.0, 2.0, 0.0, 1.0, 0.5, 1.5))

    intersection_self = bbox_a.intersection(bbox_a)
    assert intersection_self is bbox_a

    assert bbox_a.intersection(bbox_c) is None


def test_bounding_box_containment():
    outer = BoundingBox3D((0.0, 3.0, 0.0, 3.0, 0.0, 3.0))
    inner = BoundingBox3D((1.0, 2.0, 1.0, 2.0, 1.0, 2.0))
    overlapping = BoundingBox3D((2.5, 3.5, 2.5, 3.5, 2.5, 3.5))

    assert inner.is_contained_in(outer)
    assert outer.contains(inner)

    assert not outer.is_contained_in(inner)
    assert not inner.contains(outer)

    assert not overlapping.is_contained_in(outer)
    assert not outer.contains(overlapping)


def test_bounding_box_wrong_number_of_values():
    with pytest.raises(ValueError, match="Expected exactly 6 values"):
        BoundingBox3D((0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="Expected exactly 6 values"):
        BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 99.0))
    with pytest.raises(ValueError, match="Expected exactly 6 values"):
        BoundingBox3D(())


def test_bounding_box_accepts_other_iterables():
    from_list = BoundingBox3D([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    from_tuple = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    from_gen = BoundingBox3D(float(x) for x in range(6))

    assert from_list == from_tuple
    assert from_gen.min_x == 0.0
    assert from_gen.max_z == 5.0


def test_bounding_box_type_guards():
    bbox = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="'other' must be a BoundingBox3D"):
        bbox.intersects("not a bbox")
    with pytest.raises(ValueError, match="'other' must be a BoundingBox3D"):
        bbox.intersection("not a bbox")
    with pytest.raises(ValueError, match="'other' must be a BoundingBox3D"):
        bbox.is_contained_in("not a bbox")
    with pytest.raises(ValueError, match="'other' must be a BoundingBox3D"):
        bbox.contains("not a bbox")
    with pytest.raises(ValueError, match="'other' must be a BoundingBox3D"):
        bbox.union("not a bbox")


def test_bounding_box_self_identity_containment():
    bbox = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    assert bbox.is_contained_in(bbox)
    assert bbox.contains(bbox)


def test_bounding_box_is_frozen():
    bbox = BoundingBox3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    with pytest.raises(AttributeError):
        bbox._values = (9.0, 9.0, 9.0, 9.0, 9.0, 9.0)
