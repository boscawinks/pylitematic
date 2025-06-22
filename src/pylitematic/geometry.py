from __future__ import annotations

from dataclasses import dataclass
import nbtlib
import numpy as np


@dataclass(frozen=True)
class Vec3i:
    _a: int
    _b: int
    _c: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_a", self._to_int(self._a))
        object.__setattr__(self, "_b", self._to_int(self._b))
        object.__setattr__(self, "_c", self._to_int(self._c))

    def __str__(self) -> str:
        return str(list(self))

    def __len__(self) -> int:
        return 3

    def __add__(self, other) -> Vec3i:
        arr = np.array(self)
        other = self._to_array(other)
        try:
            result = arr + other
        except Exception:
            return NotImplemented
        return type(self)(*result.astype(int))

    def __radd__(self, other) -> Vec3i:
        return self.__add__(other)

    def __sub__(self, other) -> Vec3i:
        arr = np.array(self)
        other = self._to_array(other)
        try:
            result = arr - other
        except Exception:
            return NotImplemented
        return type(self)(*result.astype(int))

    def __rsub__(self, other) -> Vec3i:
        arr = np.array(self)
        try:
            result = other - arr
        except Exception:
            return NotImplemented
        return type(self)(*result.astype(int))

    def __mul__(self, other) -> Vec3i:
        arr = np.array(self)
        other = self._to_array(other)
        try:
            result = arr * other
        except Exception:
            return NotImplemented
        return type(self)(*result.astype(int))

    def __rmul__(self, other) -> Vec3i:
        return self.__mul__(other)

    def __floordiv__(self, other) -> Vec3i:
        arr = np.array(self)
        other = self._to_array(other)
        try:
            result = arr // other
        except Exception:
            return NotImplemented
        return type(self)(*result.astype(int))

    def __rfloordiv__(self, other) -> Vec3i:
        arr = np.array(self)
        try:
            result = other // arr
        except Exception:
            return NotImplemented
        return type(self)(*result.astype(int))

    def __neg__(self) -> Vec3i:
        return type(self)(-self._a, -self._b, -self._c)

    def __getitem__(self, index: int) -> int:
        return self.to_tuple()[index]

    def __iter__(self):
        return iter(self.to_tuple())

    def __abs__(self) -> Vec3i:
        return type(self)(*np.abs(self))

    def __lt__(self, other):
        return np.array(self) < self._to_array(other)

    def __le__(self, other):
        return np.array(self) <= self._to_array(other)

    def __gt__(self, other):
        return np.array(self) > self._to_array(other)

    def __ge__(self, other):
        return np.array(self) >= self._to_array(other)

    def __eq__(self, other):
        return np.array(self) == self._to_array(other)

    def __ne__(self, other):
        return np.array(self) != self._to_array(other)

    def __array__(self, dtype: type | None = None, copy: bool = True):
        arr = np.array([self._a, self._b, self._c], dtype=dtype)
        if copy:
            return arr.copy()
        else:
            return arr

    def _to_array(self, other):
        if isinstance(other, Vec3i):
            return np.array(other)
        else:
            return other

    @staticmethod
    def _to_int(value) -> int:
        if isinstance(value, (int, np.integer)):
            return int(value)
        elif isinstance(value, float):
            if value.is_integer():
                return int(value)
        raise TypeError(
            f"{type(value).__name__} value {value!r} is not"
            " int, numpy integer, or whole float")

    def to_tuple(self) -> tuple[int, int, int]:
        return (self._a, self._b, self._c)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int]) -> Vec3i:
        return cls(*t)

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

    def end(self, axis: tuple[int,...] | int | None = None) -> BlockPosition:
        limit = self - np.sign(self)

        if axis is None:
            return BlockPosition(*limit)

        if not isinstance(axis, tuple):
            axis = (axis, )

        ret = np.zeros_like(limit, dtype=int)
        for ax in axis:
            ret[ax] = limit[ax]
        return BlockPosition(*ret)
