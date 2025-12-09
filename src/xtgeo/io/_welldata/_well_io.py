"""I/O of well data"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from xtgeo.common.log import null_logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


logger = null_logger(__name__)


@dataclass(frozen=True)
class WellLog:
    """Immutable container for a single well log.

    A well log represents a continuous or discrete measurement along a wellbore.

    Attributes:
        name: The name of the log (e.g., "GR", "PHIT", "FACIES")
        values: NumPy array containing the log values. Undefined/missing values
            should be represented as np.nan for continuous logs or a specific
            undefined integer value for discrete logs.
        is_discrete: If True, this is a discrete/categorical log; if False,
            it's a continuous log.
        code_names: For discrete logs: mapping from codes (integers) to names
            Example: {0: "SHALE", 1: "SAND", 2: "LIMESTONE"}
            For continuous logs: optional metadata tuple (e.g., ("CONT", "UNK", "lin"))

    """

    name: str
    values: npt.NDArray[np.float64]
    is_discrete: bool = False
    code_names: dict[int, str] | tuple | None = None

    def __post_init__(self) -> None:
        """Validate the well log data."""
        if (
            self.is_discrete
            and self.code_names is not None
            and isinstance(self.code_names, dict)
            and not all(isinstance(k, (int, np.integer)) for k in self.code_names)
        ):
            raise ValueError("code_names keys must be integers for discrete logs")


@dataclass(frozen=True)
class WellData:
    """Immutable internal data container for well data read/write.

    This dataclass stores all the essential information about a well, including
    its position, trajectory (survey), and associated well logs. The design is
    immutable to ensure data integrity during coming I/O operations.

    Attributes:
        name: Well name (e.g., "31/2-E-4 AH")
        xpos: X-coordinate of the well header position
        ypos: Y-coordinate of the well header position
        zpos: Z-coordinate of the well header position (typically RKB, if present)
        survey_x: NumPy array of X-coordinates along the wellbore trajectory
            (float64)
        survey_y: NumPy array of Y-coordinates along the wellbore trajectory
            (float64)
        survey_z: NumPy array of Z-coordinates (typically TVD/TVDSS) along
            trajectory (float64)
        logs: Tuple of WellLog objects containing well log data. Use tuple for
            immutability. Empty tuple if no logs. Log values are stored as float64
            by default (TODO: consider change to float32 for less memory use.)

    Notes:
        - All NumPy arrays should have the same length (number of survey points)
        - Undefined values in continuous logs should be np.nan
        - Logs are stored as a tuple to maintain immutability
    """

    name: str
    xpos: float
    ypos: float
    zpos: float
    survey_x: npt.NDArray[np.float64]
    survey_y: npt.NDArray[np.float64]
    survey_z: npt.NDArray[np.float64]
    logs: tuple[WellLog, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate well data consistency."""

        n_survey = len(self.survey_x)
        if len(self.survey_y) != n_survey or len(self.survey_z) != n_survey:
            raise ValueError(
                f"Survey arrays must have the same length. Got X:{len(self.survey_x)}, "
                f"Y:{len(self.survey_y)}, Z:{len(self.survey_z)}"
            )

        for log in self.logs:
            if len(log.values) != n_survey:
                raise ValueError(
                    f"Log '{log.name}' has {len(log.values)} values, but survey "
                    f"has {n_survey} points"
                )

    @property
    def n_records(self) -> int:
        """Return the number of survey records/points."""
        return len(self.survey_x)

    @property
    def log_names(self) -> tuple[str, ...]:
        """Return tuple of all log names."""
        return tuple(log.name for log in self.logs)

    def get_log(self, name: str) -> WellLog | None:
        """Get a well log by name.

        Args:
            name: Name of the log to retrieve

        Returns:
            WellLog object if found, None otherwise
        """
        for log in self.logs:
            if log.name == name:
                return log
        return None

    def get_continuous_logs(self) -> tuple[WellLog, ...]:
        """Return tuple of all continuous logs."""
        return tuple(log for log in self.logs if not log.is_discrete)

    def get_discrete_logs(self) -> tuple[WellLog, ...]:
        """Return tuple of all discrete logs."""
        return tuple(log for log in self.logs if log.is_discrete)

    def to_dataframe(
        self,
        lognames: list[str] | None = None,
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
    ) -> "pd.DataFrame":
        """Return survey and selected logs as a pandas DataFrame.

        Args:
            lognames: List of log names to include. If None, all logs are included.
            xname: Column name for X coordinates (default: "X_UTME")
            yname: Column name for Y coordinates (default: "Y_UTMN")
            zname: Column name for Z coordinates (default: "Z_TVDSS")

        Returns:
            pandas.DataFrame with survey coordinates and selected logs

        Raises:
            ValueError: If a requested log name is not found

        Example:
            >>> df = well.to_dataframe(lognames=["GR", "PHIT"])
            >>> df = well.to_dataframe()  # All logs
        """
        import pandas as pd

        data = {
            xname: self.survey_x,
            yname: self.survey_y,
            zname: self.survey_z,
        }

        if lognames is None:
            # Include all logs
            for log in self.logs:
                data[log.name] = log.values
        else:
            # Include only specified logs
            for logname in lognames:
                found_log = self.get_log(logname)
                if found_log is None:
                    available = ", ".join(self.log_names)
                    raise ValueError(
                        f"Log '{logname}' not found. Available logs: {available}"
                    )
                data[logname] = found_log.values

        return pd.DataFrame(data)
