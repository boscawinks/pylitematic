from __future__ import annotations

from abc import ABC, abstractmethod
import json
import nbtlib
import re
from typing import Any


PROPERTY_NAME_REGEX: str = r"[a-z][a-z0-9_]*"
PROPERTY_NAME_PATTERN: re.Pattern = re.compile(PROPERTY_NAME_REGEX)

ENUM_VALUE_REGEX: str = r"[a-z]+(_[a-z]+)*" # snake case
ENUM_VALUE_PATTERN: re.Pattern = re.compile(ENUM_VALUE_REGEX)


class Properties(dict):

    def __init__(self, *args, **kwargs):
        props = {}
        for name, value in dict(*args, **kwargs).items():
            self.validate_name(name)
            props[name] = PropertyValue.value_factory(value)
        super().__init__(props)

    def __getitem__(self, key):
        return super().__getitem__(key).get()

    def __setitem__(self, key, value):
        if key not in self:
            self.validate_name(key)
            super().__setitem__(key, PropertyValue.value_factory(value))
        else:
            super().__getitem__(key).set(value)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Properties):
            return NotImplemented
        return sorted(self.items()) < sorted(other.items())

    def __hash__(self) -> int:
        return hash(tuple(sorted(self)))

    def __str__(self) -> str:
        props_str = [f"{n}={v}" for n, v in sorted(super().items())]
        return f"[{','.join(props_str)}]"

    def __repr__(self) -> str:
        props_reps = [f"{n}: {v!r}" for n, v in sorted(super().items())]
        return f"{type(self).__name__}({', '.join(props_reps)})"

    def get(self, key, default=None):
        value = super().get(key, None)
        if value is None:
            return default
        return value.get()

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *args, **kwargs):
        for name, value in dict(*args, **kwargs).items():
            self[name] = value

    def pop(self, key, default=None):
        value = super().pop(key, None)
        if value is None:
            return default
        return value.get()

    def popitem(self):
        name, value = super().popitem()
        return name, value.get()

    def values(self):
        for value in super().values():
            yield value.get()

    def items(self):
        for key, value in super().items():
            yield key, value.get()

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return PROPERTY_NAME_PATTERN.fullmatch(name) is not None

    @staticmethod
    def validate_name(name: str) -> None:
        if not Properties.is_valid_name(name=name):
            raise ValueError(f"Invalid property name {name!r}")

    def to_string(self) -> str:
        return str(self)

    @staticmethod
    def from_string(string: str) -> Properties:
        if string in ("", "[]"):
            return Properties()

        if not (string.startswith("[") and string.endswith("]")):
            raise ValueError(f"Invalid properties string {string!r}")
        string = string[1:-1]

        props = {}
        for prop in string.split(","):
            try:
                name, val_str = prop.split("=")
            except ValueError as exc:
                raise ValueError(f"Invalid property string {string!r}") from exc
            if name in props:
                ValueError(f"Duplicate property name {name!r}")
            props[name] = PropertyValue.from_string(string=val_str).get()

        return Properties(props)

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound(
            {name: value.to_nbt() for name, value in sorted(super().items())})

    @staticmethod
    def from_nbt(nbt: nbtlib.Compound) -> Properties:
        props = {}
        for name, value in nbt.items():
            props[name] = PropertyValue.from_nbt(nbt=value).get()
        return Properties(props)


class PropertyValue(ABC):

    __slots__ = ("_value")
    __registry: dict[type, type[PropertyValue]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        py_type = cls.python_type()
        if py_type in cls.__registry:
            raise ValueError(
                f"Duplicate Value subclass for type {py_type.__name__!r}:"
                f" {cls.__registry[py_type].__name__} vs {cls.__name__}")
        cls.__registry[py_type] = cls

    def __init__(self, value: Any) -> None:
        self.set(value)

    def __str__(self) -> str:
        return json.dumps(self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"

    def __hash__(self) -> int:
        return hash((self.__class__, self._value))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value == other._value

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, PropertyValue):
            return NotImplemented
        return self._value < other._value

    @classmethod
    @abstractmethod
    def is_valid_value(cls, value: Any) -> bool:
        ...

    @classmethod
    def validate_value(cls, value: Any) -> None:
        if not isinstance(value, cls.python_type()):
            raise TypeError(
                f"{cls.__name__} expects value of type"
                f" {cls.python_type().__name__}, got {type(value).__name__}"
                f" ({value!r})")
        if not cls.is_valid_value(value):
            raise ValueError(f"Invalid value {value!r} for {cls.__name__}")

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self.validate_value(value=value)
        self._value = self.python_type()(value)

    @classmethod
    @abstractmethod
    def python_type(cls) -> type:
        """Return the native Python type this Value corresponds to."""

    @staticmethod
    def value_factory(value: Any) -> PropertyValue:
        reg = PropertyValue.__registry
        sub_cls = reg.get(type(value))
        if sub_cls is None:
            opt_str = ", ".join(map(lambda x: x.__name__, reg))
            raise TypeError(
                f"No Value subclass registered for {type(value).__name__} value"
                f" {value!r}. Classes registered for: {opt_str}")
        return sub_cls(value)

    def to_string(self) -> str:
        return str(self)

    @staticmethod
    def from_string(string: str) -> PropertyValue:
        try:
            value = json.loads(string)
        except json.JSONDecodeError:
            value = string
        return PropertyValue.value_factory(value)

    def to_nbt(self) -> nbtlib.String:
        return nbtlib.String(self)

    @staticmethod
    def from_nbt(nbt: nbtlib.String) -> PropertyValue:
        return PropertyValue.from_string(str(nbt))


class BooleanValue(PropertyValue):

    @classmethod
    def is_valid_value(cls, value: Any) -> bool:
        return True

    @classmethod
    def python_type(cls) -> type:
        """Return the native Python type a BooleanValue corresponds to."""
        return bool


class IntegerValue(PropertyValue):

    @classmethod
    def is_valid_value(cls, value: Any) -> bool:
        return value >= 0

    @classmethod
    def python_type(cls) -> type:
        """Return the native Python type an IntegerValue corresponds to."""
        return int


class EnumValue(PropertyValue):

    def __str__(self) -> str:
        return self._value

    @classmethod
    def is_valid_value(cls, value: Any) -> bool:
        return ENUM_VALUE_PATTERN.fullmatch(value) is not None

    @classmethod
    def python_type(cls) -> type:
        """Return the native Python type an EnumValue corresponds to."""
        return str
