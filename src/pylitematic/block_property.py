from __future__ import annotations

from abc import ABC, abstractmethod
import json
from nbtlib import String
import re
from typing import Any


PROPERTY_NAME_REGEX: str = r"[a-z][a-z0-9_]*"
PROPERTY_NAME_PATTERN: re.Pattern = re.compile(PROPERTY_NAME_REGEX)


class Property():

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: Any | Value) -> None:
        if not PROPERTY_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid property name {name!r}")
        self._name = name

        if not isinstance(value, Value):
            value = Value.value_factory(value=value)
        self._value = value

    def __str__(self) -> str:
        return f"{self._name}={self._value}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name: {self._name}, value: {self._value!r})")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Property):
            return NotImplemented
        return (self.name, self.value) == (other.name, other.value)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Property):
            return NotImplemented
        return (self.name, self.value) < (other.name, other.value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value.get()

    @value.setter
    def value(self, value: Any) -> None:
        self._value.set(value)

    def to_string(self) -> str:
        return str(self)

    @staticmethod
    def from_string(string: str, value: str | None = None) -> Property:
        if value is None:
            # tread string as "name=value"
            try:
                string, value = string.split("=")
            except ValueError as exc:
                raise ValueError(f"Invalid property string {string!r}") from exc
        return Property(name=string, value=Value.from_string(value))

    def to_nbt(self) -> tuple[str, String]:
        # return Compound(Name=String(self._name), Value=self._value.to_nbt()})
        return self._name, self._value.to_nbt()

    @staticmethod
    def from_nbt(name: str, nbt: String) -> Property:
        # return Property.from_string(name=nbt["Name"], value=str(nbt["Value"]))
        return Property.from_string(string=name, value=str(nbt))


class Value(ABC):

    __slots__ = ("_value")
    __registry: dict[type, type[Value]] = {}

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

    def __repr__(self):
        return f"{type(self).__name__}({self._value!r})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return self._value == other._value

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return self._value < other._value

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        if not isinstance(value, self.python_type()):
            raise TypeError(
                f"{type(self).__name__} expects value of type"
                f" {self.python_type().__name__}, got {type(value).__name__}"
                f" ({value!r})")
        self._value = self.python_type()(value)

    @classmethod
    @abstractmethod
    def python_type(cls) -> type:
        """Return the native Python type this Value corresponds to."""

    @staticmethod
    def value_factory(value: Any) -> Value:
        sub_cls = Value.__registry.get(type(value))
        if sub_cls is None:
            opt_str = ", ".join(map(lambda x: x.__name__, Value.__registry))
            raise TypeError(
                f"No Value subclass registered for {type(value).__name__} value"
                f" {value!r}. Classes registered for: {opt_str}")
        return sub_cls(value)

    def to_string(self) -> str:
        return str(self)

    @staticmethod
    def from_string(string: str) -> Value:
        try:
            value = json.loads(string)
        except json.JSONDecodeError:
            value = string
        return Value.value_factory(value)

    def to_nbt(self) -> String:
        return String(str(self))

    @staticmethod
    def from_nbt(nbt: String) -> Value:
        return Value.from_string(str(nbt))


class BooleanValue(Value):

    @classmethod
    def python_type(cls) -> type:
        """Return the native Python type a BooleanValue corresponds to."""
        return bool


class IntegerValue(Value):

    @classmethod
    def python_type(cls) -> type:
        """Return the native Python type an IntegerValue corresponds to."""
        return int


class EnumValue(Value):

    def __str__(self) -> str:
        return self._value

    @classmethod
    def python_type(cls) -> type:
        """Return the native Python type an EnumValue corresponds to."""
        return str
