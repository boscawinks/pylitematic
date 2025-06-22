import pytest
from nbtlib import String
from typing import Any

from pylitematic.block_property import (
    BooleanValue,
    EnumValue,
    IntegerValue,
    PropertyValue,
)


def test_value():
    # TODO
    # * test sorting of values

    def check_value(
        value: PropertyValue,
        target: Any,
        string: str,
        nbt: String,
        typ: type,
        repr_str: str,
    ) -> None:
        assert value.get() == target
        value.set(target)
        assert isinstance(value, typ)
        assert value.get() == target
        assert repr(value) == repr_str
        assert str(value) == string
        assert value.to_string() == string
        assert value.to_nbt() == nbt

    proto_values = [
        (BooleanValue, True, "true", String("true"), "BooleanValue(True)"),
        (EnumValue, "north", "north", String("north"), "EnumValue('north')"),
        (IntegerValue, 42, "42", String("42"), "IntegerValue(42)"),
    ]

    for typ, target, string, nbt, repr_str in proto_values:
        value = typ(target)
        check_value(value, target, string, nbt, typ, repr_str)
        value = PropertyValue.value_factory(target)
        check_value(value, target, string, nbt, typ, repr_str)
        value = PropertyValue.from_string(string)
        check_value(value, target, string, nbt, typ, repr_str)
        value = PropertyValue.from_nbt(nbt)
        check_value(value, target, string, nbt, typ, repr_str)

    with pytest.raises(TypeError):
        PropertyValue.value_factory(value=3.14)
