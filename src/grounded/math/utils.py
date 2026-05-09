import math


def normalize_degrees(degrees: float) -> float:
    """
    Normalize a degrees value to be in range [-180 180].

    Parameters:
        degrees: An arbitrary degrees value.

    Returns:
        A normalized degrees value.
    """
    return degrees - 360.0 * math.floor((degrees + 180.0) / 360.0)