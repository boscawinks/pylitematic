import pytest
from nbtlib import String
from typing import Any

from pylitematic.block_property import (
    BooleanValue,
    EnumValue,
    IntegerValue,
    Property,
    Value,
)


def test_value():
    # TODO
    # * test sorting of values

    def check_value(
        value: Value,
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
        value = Value.value_factory(target)
        check_value(value, target, string, nbt, typ, repr_str)
        value = Value.from_string(string)
        check_value(value, target, string, nbt, typ, repr_str)
        value = Value.from_nbt(nbt)
        check_value(value, target, string, nbt, typ, repr_str)

    with pytest.raises(TypeError):
        Value.value_factory(value=3.14)


def test_property():
    # TODO
    # * test sorting of properties

    def check_prop(
        prop: Property,
        name: str,
        target: Any,
        string: str,
        nbt: tuple[str, String],
    ) -> None:
        assert prop.name == name
        assert prop.value == target
        prop.value = target
        assert str(prop) == string
        assert prop.to_string() == string
        assert prop.to_nbt() == nbt

    proto_props = [
        ("enabled", True, "true", String("true"), "enabled=true"),
        ("facing", "north", "north", String("north"), "facing=north"),
        ("age", 42, "42", String("42"), "age=42"),
    ]

    for name, target, string, nbt, prop_string in proto_props:
        prop = Property(name=name, value=target)
        check_prop(prop, name, target, prop_string, (name, nbt))
        prop = Property.from_string(string=name, value=string)
        check_prop(prop, name, target, prop_string, (name, nbt))
        prop = Property.from_string(string=prop_string)
        check_prop(prop, name, target, prop_string, (name, nbt))
        prop = Property.from_nbt(name=name, nbt=nbt)
        check_prop(prop, name, target, prop_string, (name, nbt))

    with pytest.raises(TypeError):
        Property(name="float", value=3.14)

    with pytest.raises(ValueError):
        Property(name="Uppercase", value=1)
