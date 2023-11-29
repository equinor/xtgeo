from math import degrees, radians


def _deg_angle2azimuth(deg: float) -> float:
    """Converts an angle from X-axis anti-clockwise orientation
    to Y-axis clockwise azimuth.
    """
    return (-deg + 90) % 360


def _deg_azimuth2angle(deg: float) -> float:
    """Converts azimuth from Y-axis clockwise orientation to X-axis
    anti-clockwise angle.
    """
    return (450 - deg) % 360


def _rad_angle2azimuth(rad: float) -> float:
    """Converts an angle from X-axis anti-clockwise orientation to Y-axis clockwise
    azimuth in radians.
    """
    return radians(_deg_angle2azimuth(degrees(rad)))


def _rad_azimuth2angle(deg: float) -> float:
    """Converts azimuth from Y-axis clockwise orientation to X-axis anti-clockwise
    angle in radians.
    """
    return radians(_deg_azimuth2angle(degrees(deg)))
