from __future__ import annotations

from nbtlib import String
import re
from typing import Any


NAMESPACE_REGEX: str = r"[a-z0-9_.-]+"
NAMESPACE_PATTERN: re.Pattern = re.compile(NAMESPACE_REGEX)
DEFAULT_NAMESPACE: str = "minecraft"

PATH_REGEX: str = r"[a-z0-9_.-][a-z0-9_./-]*"
PATH_PATTERN: re.Pattern = re.compile(PATH_REGEX)

LOCATION_PATTERN: re.Pattern = re.compile(
    rf"(?:(?P<namespace>{NAMESPACE_REGEX})?\:)?(?P<path>{PATH_REGEX})")

class ResourceLocation:

    __slots__ = ("_path", "_namespace")

    def __init__(self, path: str, namespace: str | None = None) -> None:
        if not PATH_PATTERN.fullmatch(path):
            raise ValueError(f"Invalid resource location path {path!r}")
        self._path = path

        if namespace is None:
            namespace = DEFAULT_NAMESPACE
        if not NAMESPACE_PATTERN.fullmatch(namespace):
            raise ValueError(
                f"Invalid resource location namespace {namespace!r}")
        self._namespace = namespace

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (ResourceLocation, str)):
            return NotImplemented
        return str(self) == str(other)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ResourceLocation):
            return NotImplemented
        return str(self) < str(other)

    def __str__(self) -> str:
        return f"{self.namespace}:{self.path}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"namespace: {self.namespace}, path: {self.path})")

    @property
    def path(self) -> str:
        return self._path

    @property
    def namespace(self) -> str:
        return self._namespace

    def to_string(self) -> str:
        return str(self)

    @staticmethod
    def from_string(string: str) -> ResourceLocation:
        match = LOCATION_PATTERN.fullmatch(string)
        if not match:
            raise ValueError(f"Invalid resource location string {string!r}")

        namespace = match.group("namespace")
        path = match.group("path")

        namespace = None if namespace == "" else namespace
        return ResourceLocation(path=path, namespace=namespace)

    def to_nbt(self) -> String:
        return String(self)

    @staticmethod
    def from_nbt(nbt: String) -> ResourceLocation:
        return ResourceLocation.from_string(str(nbt))
