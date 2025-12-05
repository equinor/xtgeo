from typing import Any, Protocol

from typing_extensions import Self


class ValidateProtocol(Protocol):
    """Protocol for pre-validation, e.g. of data read from file."""

    def pre_validate(cls: Self, data: dict[str, Any], fileref_errmsg: str) -> None:
        """
        Validate data prior to creating object.

        Args:
            data: Data to validate
            fileref_errmsg: Error context for meaningful error messages

        Raises:
            ValueError: If data is invalid
        """
