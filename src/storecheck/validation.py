"""Shared validation for numeric public settings."""

import math


def positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    """Require an integer count within its allowed range.

    Args:
        name: Parameter name used in errors.
        value: Supplied count.
        allow_zero: Whether zero is permitted.

    Raises:
        ValueError: If the count is not an integer in the permitted range.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(not allow_zero)
    ):
        raise ValueError(f"{name} must be an integer >= {int(not allow_zero)}")


def fraction(name: str, value: float) -> None:
    """Require a finite fraction between zero and one inclusive.

    Args:
        name: Parameter name used in errors.
        value: Supplied fraction.

    Raises:
        ValueError: If the value is not a finite fraction.
    """
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
