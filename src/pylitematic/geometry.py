from __future__ import annotations

from dataclasses import dataclass
import enum
import nbtlib
import numpy as np
from typing import Iterator


@dataclass(frozen=True)
class Vec3i:
    _a: int
    _b: int
    _c: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_a", int(self._a))
        object.__setattr__(self, "_b", int(self._b))
        object.__setattr__(self, "_c", int(self._c))

    def __getitem__(self, index: int) -> int:
        return tuple(self)[index]

    def __str__(self) -> str:
        return str(list(self))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(a={self._a}, b={self._b}, c={self._c})")

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[int]:
        return iter((self._a, self._b, self._c))

    def __neg__(self) -> Vec3i:
        return type(self)(*(-i for i in self))

    def __abs__(self) -> Vec3i:
        return type(self)(*(abs(i) for i in self))

    def __array__(self, dtype: type | None = None, copy: bool = True):
        arr = np.array(tuple(self), dtype=dtype)
        return arr.copy() if copy else arr

    def __add__(self, other) -> Vec3i:
        return type(self)(*(np.array(self) + other))

    def __radd__(self, other) -> Vec3i:
        return self.__add__(other)

    def __sub__(self, other) -> Vec3i:
        return self.__add__(-other)

    def __rsub__(self, other) -> Vec3i:
        return -self.__sub__(other)

    def __mul__(self, other) -> Vec3i:
        return type(self)(*(np.array(self) * other))

    def __rmul__(self, other) -> Vec3i:
        return self.__mul__(other)

    def __floordiv__(self, other) -> Vec3i:
        return type(self)(*(np.array(self) // other))

    def __rfloordiv__(self, other) -> Vec3i:
        return type(self)(*(other // np.array(self)))

    def __truediv__(self, other) -> Vec3i:
        return self.__floordiv__(other)

    def __rtruediv__(self, other) -> Vec3i:
        return self.__rfloordiv__(other)

    def __mod__(self, other) -> Vec3i:
        return type(self)(*(np.array(self) % other))

    def __rmod__(self, other) -> Vec3i:
        return type(self)(*(other % np.array(self)))

    def __eq__(self, other):
        return np.array(self) == other

    def __ne__(self, other):
        return np.invert(self.__eq__(other))

    def __lt__(self, other):
        return np.array(self) < other

    def __le__(self, other):
        return np.array(self) <= other

    def __gt__(self, other):
        return np.array(self) > other

    def __ge__(self, other):
        return np.array(self) >= other

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
            "x": nbtlib.Int(self._a),
            "y": nbtlib.Int(self._b),
            "z": nbtlib.Int(self._c),
        })

    @classmethod
    def from_nbt(cls, nbt: nbtlib.Compound) -> Vec3i:
        return cls(int(nbt["x"]), int(nbt["y"]), int(nbt["z"]))


@dataclass(frozen=True)
class BlockPosition(Vec3i):

    @property
    def x(self) -> int:
        return self._a

    @property
    def y(self) -> int:
        return self._b

    @property
    def z(self) -> int:
        return self._c

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x={self.x}, y={self.y}, z={self.z})"


class Direction(BlockPosition, enum.Enum):
    NORTH = 0,  0, -1
    SOUTH = 0,  0,  1
    WEST = -1,  0,  0
    EAST =  1,  0,  0
    UP =    0,  1,  0
    DOWN =  0, -1,  0


@dataclass(frozen=True)
class Size3D(Vec3i):

    @property
    def width(self) -> int:
        return self._a

    @property
    def height(self) -> int:
        return self._b

    @property
    def length(self) -> int:
        return self._c

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"width={self.width}, height={self.height}, length={self.length})")

    @property
    def volume(self) -> int:
        return abs(self.width * self.height * self.length)

    @property
    def limit(self) -> BlockPosition:
        return BlockPosition(*(self - self.sign))

    @property
    def sign(self) -> BlockPosition:
        return BlockPosition(*np.sign(self))
