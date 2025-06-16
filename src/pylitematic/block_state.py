from __future__ import annotations

from typing import Any, Iterable

from .resource_location import ResourceLocation
from .block_property import Property


class BlockState:
    # TODO:
    # * make immutable
    # * make hashable
    # * create methods to init copies with different properties
    # * __eq__ for strings

    __slots__ = ("_id", "_props")

    def __init__(self, id: ResourceLocation, **kwargs) -> None:
        self._id = id
        self._props: dict[str, Property] = {}
        for name, value in kwargs.items():
            self._props[name] = Property.property_factory(name, value)

    def __contains__(self, name: str) -> bool:
        return name in self._props

    def __getitem__(self, name: str) -> Any:
        try:
            return self._props[name].value
        except KeyError as exc:
            raise KeyError(
                f"{type(self).__name__} '{self}' does not"
                f" have {name!r} property") from exc

    def __getattr__(self, name: str) -> Any:
        return self[name]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BlockState):
            return NotImplemented
        return (self.id, self._props) == (other.id, other._props)
        # return str(self) == str(other)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BlockState):
            return NotImplemented
        # return (self.id, str(self._props)) < (other.id, str(other._props))
        return str(self) < str(other)

    def __str__(self) -> str:
        if not self._props:
            return str(self.id)
        props_str = ",".join(map(str, [x for _, x in self.properties()]))
        return f"{self.id}[{props_str}]"

    @property
    def id(self) -> ResourceLocation:
        return self._id

    def properties(self) -> Iterable[tuple[str, Any]]:
        return sorted(self._props.items())

    @classmethod
    def from_string(cls, string: str) -> BlockState:
        ...


bs = BlockState("minecraft:stone", attached=False, age=42, attachment="floor")
BS = BlockState("minecraft:stone", attachment="floor", age=42, attached=True)
AIR = BlockState("minecraft:air")

print(bs)
print(BS)
print(AIR)

print(f"[{', '.join(list(map(str, sorted([bs, BS, AIR]))))}]")

print(bs == BS)
print(bs["attachment"], bs.age)
