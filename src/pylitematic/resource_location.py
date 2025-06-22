from __future__ import annotations

from dataclasses import dataclass
import nbtlib
import re
from typing import Any


NAMESPACE_REGEX: str = r"[a-z0-9_.-]+"
NAMESPACE_PATTERN: re.Pattern = re.compile(NAMESPACE_REGEX)
DEFAULT_NAMESPACE: str = "minecraft"

PATH_REGEX: str = r"[a-z0-9_.-][a-z0-9_./-]*"
PATH_PATTERN: re.Pattern = re.compile(PATH_REGEX)

LOCATION_PATTERN: re.Pattern = re.compile(
    rf"(?:(?P<namespace>{NAMESPACE_REGEX})?\:)?(?P<path>{PATH_REGEX})")

@dataclass(frozen=True, order=True)
class ResourceLocation:

    path: str
    namespace: str = ""

    def __post_init__(self) -> None:
        if not PATH_PATTERN.fullmatch(self.path):
            raise ValueError(f"Invalid resource location path {self.path!r}")

        if not self.namespace:
            object.__setattr__(self, "namespace", DEFAULT_NAMESPACE)
        elif not NAMESPACE_PATTERN.fullmatch(self.namespace):
            raise ValueError(
                f"Invalid resource location namespace {self.namespace!r}")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ResourceLocation):
            return (self.namespace, self.path) == (other.namespace, other.path)
        elif isinstance(other, str):
            try:
                other = ResourceLocation.from_string(other)
                return self == other
            except ValueError:
                return False
        else:
            return NotImplemented

    def __str__(self) -> str:
        return f"{self.namespace}:{self.path}"

    def to_string(self) -> str:
        return str(self)

    @staticmethod
    def from_string(string: str) -> ResourceLocation:
        match = LOCATION_PATTERN.fullmatch(string)
        if not match:
            raise ValueError(f"Invalid resource location string {string!r}")

        namespace = match.group("namespace")
        path = match.group("path")

        return ResourceLocation(path=path, namespace=namespace)

    def to_nbt(self) -> nbtlib.String:
        return nbtlib.String(self)

    @staticmethod
    def from_nbt(nbt: nbtlib.String) -> ResourceLocation:
        return ResourceLocation.from_string(str(nbt))


class BlockId(ResourceLocation):
    ...
