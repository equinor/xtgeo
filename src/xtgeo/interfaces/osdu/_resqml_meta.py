# -*- coding: utf-8 -*-
"""Helpers for attaching/retrieving RESQML metadata on xtgeo objects.

Not all xtgeo types have a `metadata` attribute (e.g., Points, Polygons don't).
These helpers provide a uniform way to store/retrieve RESQML provenance via:
  1. ``obj.metadata.freeform["resqml"]`` — if the object has
     metadata (Grid, Surface, etc.)
  2. ``obj._resqml_meta`` — fallback attribute for objects
     without metadata (Points, Polygons)

This avoids modifying existing xtgeo classes while still
carrying RESQML identity (UUID, CRS, property kind, etc.)
through geomodelling workflows.
"""

from __future__ import annotations

from typing import Any, Dict


def _set_resqml_meta(obj: Any, meta: Dict[str, Any]) -> None:
    """Attach RESQML metadata to an xtgeo object.

    Uses `metadata.freeform` if available, otherwise sets `_resqml_meta` attribute.
    """
    md = getattr(obj, "metadata", None)
    if md is not None and hasattr(md, "freeform"):
        md.freeform["resqml"] = meta
    else:
        object.__setattr__(obj, "_resqml_meta", meta)


def _get_resqml_meta(obj: Any) -> Dict[str, Any]:
    """Retrieve RESQML metadata from an xtgeo object.

    Returns empty dict if no metadata is attached.
    """
    md = getattr(obj, "metadata", None)
    if md is not None and hasattr(md, "freeform"):
        return md.freeform.get("resqml", {})
    return getattr(obj, "_resqml_meta", {})
