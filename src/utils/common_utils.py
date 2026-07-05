from enum import Enum
from typing import TypeVar

E = TypeVar("E", bound=Enum)


def enum_from_value(enum_type: type[E], value: object) -> E:
    try:
        return enum_type(value)
    except ValueError as exc:
        valid_values = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(
            f"{value!r} is not a valid {enum_type.__name__}. "
            f"Expected one of: {valid_values}"
        ) from exc
