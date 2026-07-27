"""Helpers for handling deprecated (renamed) arguments in the ResInsight API."""

from __future__ import annotations

import warnings
from typing import TypeVar

T = TypeVar("T")


def resolve_deprecated_alias(
    new_value: T | None,
    old_value: T | None,
    *,
    new_name: str,
    old_name: str,
    stacklevel: int = 3,
) -> T | None:
    """Return the effective value for an argument that was renamed.

    ``None`` marks "not provided" (the ResInsight case/property arguments are
    never legitimately ``None``). Emits a :class:`DeprecationWarning` when the
    deprecated (old) name is used, and raises :class:`TypeError` if both names
    are supplied at once.

    Args:
        new_value: Value passed via the current argument name (or ``None``).
        old_value: Value passed via the deprecated argument name (or ``None``).
        new_name: The current argument name (used in messages).
        old_name: The deprecated argument name (used in messages).
        stacklevel: Stack level forwarded to :func:`warnings.warn` so the warning
            points at the caller of the public function (default 3).
    """
    if old_value is not None:
        if new_value is not None:
            raise TypeError(
                f"Got values for both '{new_name}' and its deprecated alias "
                f"'{old_name}'; please pass only '{new_name}'."
            )
        warnings.warn(
            f"The '{old_name}' argument is deprecated and will be removed in a "
            f"future version; use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return old_value
    return new_value
