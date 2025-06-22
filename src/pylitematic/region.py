from __future__ import annotations

from bitpacking import bitpack, bitunpack
from functools import cached_property
import nbtlib
import numpy as np
import twos
from typing import Iterator

from .block_state import BlockState
from .geometry import BlockPosition, Size3D
from .resource_location import ResourceLocation


AIR = BlockState("air")


class Region:
    def __init__(
        self,
        size: tuple[int, int, int] | Size3D,
        origin: tuple[int, int, int] | BlockPosition = (0, 0, 0),
    ):
        self._origin: BlockPosition = BlockPosition(*origin)
        self._size: Size3D = Size3D(*size)

        self._palette: list[BlockState] = [AIR]
        self._palette_map: dict[BlockState, int] = {AIR: 0}
        self._blocks = np.zeros(abs(self._size), dtype=int)

        self._entities: list[nbtlib.Compound] = []
        self._tile_entities: list[nbtlib.Compound] = []
        self._block_ticks: list[nbtlib.Compound] = []
        self._fluid_ticks: list[nbtlib.Compound] = []

    def __contains__(self, item) -> bool:
        if isinstance(item, BlockPosition):
            return all(self.lower <= item) and all(item <= self.upper)
        elif isinstance(item, BlockState):
            index = self._palette_map.get(item)
            if index is None:
                return False
            return np.any(self._blocks == index)
        elif isinstance(item, ResourceLocation):
            return any(
                (bs.id == item and np.any(self._blocks == idx))
                for bs, idx in self._palette_map.items())
        else:
            return False

    def _expand_index(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        ndim = self._blocks.ndim
        result = []
        for item in index:
            if item is Ellipsis:
                result.extend([slice(None)] * (ndim - len(index) + 1))
            else:
                result.append(item)
        while len(result) < ndim:
            result.append(slice(None))
        return tuple(result)

    def _to_internal(self, pos):
        index = []
        for i, item in enumerate(pos):
            offset = self.lower[i]
            if isinstance(item, int):
                index.append(item - offset)
            elif isinstance(item, slice):
                start = item.start - offset if item.start is not None else None
                stop = item.stop - offset if item.stop is not None else None
                index.append(slice(start, stop, item.step))
            else:
                index.append(item)
        return tuple(index)

    def _from_internal(self, index: tuple[int, int, int]) -> BlockPosition:
        return self.lower + index

    def _key_to_index(self, key):
        if isinstance(key, BlockPosition):
            index = tuple(key)
        else:
            index = self._expand_index(key)
        return self._to_internal(index)

    def __getitem__(self, key):
        index = self._key_to_index(key)
        indices = self._blocks[index]
        if np.isscalar(indices):
            return self._palette[indices]

        return np.array(self._palette, dtype=object)[indices]

    def __setitem__(self, key, value) -> None:
        index = self._key_to_index(key)

        if isinstance(value, list):
            value = np.array(value, dtype=object)

        if isinstance(value, BlockState):
            # assign single BlockState to slice
            if value not in self._palette_map:
                self._palette_map[value] = len(self._palette)
                self._palette.append(value)
            self._blocks[index] = self._palette_map[value]

        elif isinstance(value, np.ndarray):
            if value.shape != self._blocks[index].shape:
                raise ValueError(
                    "Shape mismatch between assigned array and target slice")

            # look up (or add) indices for all BlockStates
            unique_states, xdi = np.unique(value, return_inverse=True)
            idx = []
            for state in unique_states:
                if state not in self._palette_map:
                    self._palette_map[state] = len(self._palette)
                    self._palette.append(state)
                idx.append(self._palette_map[state])
            index_array = np.array(idx, dtype=int)[xdi].reshape(value.shape)
            self._blocks[index] = index_array
        else:
            raise TypeError(
                "Value must be a BlockState or a list of BlockStates")

    def compact_palette(self) -> None:
        idx = np.unique(self._blocks)
        # always include minecraft:air in a palette
        if 0 not in idx:
            idx = np.insert(idx, 0, 0)
        index_map = {old: new for new, old in enumerate(idx)}

        # compacted palette and mapping
        palette = np.array(self._palette, dtype=object)[idx].tolist()
        palette_map = {res: idx for idx, res in enumerate(palette)}

        lookup = np.full(max(index_map) + 1, -1, dtype=int)
        for old, new in index_map.items():
            lookup[old] = new
        self._blocks = lookup[self._blocks]

        self._palette = palette
        self._palette_map = palette_map

    def _bits_per_state(self) -> int:
        return max(2, (len(self._palette) - 1).bit_length())

    def _decode_block_states(
        self,
        data: nbtlib.LongArray,
    ) -> np.ndarray[int]:
        states = bitunpack(
            chunks=[twos.to_unsigned(x, 64) for x in data],
            field_width=self._bits_per_state(),
            chunk_width=64,
        )
        states = list(states)[:self.volume]
        shape = (abs(self.height), abs(self.length), abs(self.width))
        states = np.asarray(states, dtype=int).reshape(shape) # y,z,x
        return states.transpose(2, 0, 1) # x,y,z

    def _encode_block_states(self) -> nbtlib.LongArray:
        states = self._blocks.transpose(1, 2, 0).ravel() # x,y,z to y,z,x
        chunks = bitpack(
            states.tolist(),
            field_width=self._bits_per_state(),
            chunk_width=64,
        )
        return nbtlib.LongArray([twos.to_signed(x, 64) for x in chunks])

    @staticmethod
    def inclusive_end(
        size: Size3D,
        pos: BlockPosition = BlockPosition(0, 0, 0),
    ) -> BlockPosition:
        return pos + (size - np.sign(size))
        # return pos + np.where(size > 0, size - 1, size + 1)

    @property
    def size(self) -> Size3D:
        return self._size

    @cached_property
    def sign(self) -> Size3D:
        return Size3D(*np.sign(self._size))

    @property
    def origin(self) -> BlockPosition:
        return self._origin

    @cached_property
    def limit(self) -> BlockPosition:
        return self.inclusive_end(pos=self._origin, size=self._size)

    @cached_property
    def start(self) -> BlockPosition:
        return BlockPosition(0, 0, 0)

    @cached_property
    def end(self) -> BlockPosition:
        return self.inclusive_end(pos=self.start, size=self._size)

    @property
    def width(self) -> int:
        return self._size.width

    @property
    def height(self) -> int:
        return self._size.height

    @property
    def length(self) -> int:
        return self._size.length

    @property
    def volume(self) -> int:
        return np.prod(self.shape).item()

    @property
    def block_count(self) -> int:
        # TODO: Add filter BlockState and rename to count()
        return np.count_nonzero(self._blocks)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._blocks.shape

    @cached_property
    def lower(self) -> BlockPosition:
        return BlockPosition(*np.min((self.start, self.end), axis=0))

    @cached_property
    def upper(self) -> BlockPosition:
        return BlockPosition(*np.max((self.start, self.end), axis=0))

    @cached_property
    def bounds(self) -> tuple[BlockPosition, BlockPosition]:
        return self.lower, self.upper

    @cached_property
    def global_lower(self) -> BlockPosition:
        return BlockPosition(*np.min((self.origin, self.limit), axis=0))

    @cached_property
    def global_upper(self) -> BlockPosition:
        return BlockPosition(*np.max((self.origin, self.limit), axis=0))

    @cached_property
    def global_bounds(self) -> tuple[BlockPosition, BlockPosition]:
        return self.global_lower, self.global_upper

    def blocks(
        self,
        include: BlockState | list[BlockState] | None = None,
        exclude: BlockState | list[BlockState] | None = None,
        ignore_props: bool = False,
    ) -> Iterator[tuple[BlockPosition, BlockState]]:
        if isinstance(include, BlockState):
            include = [include]
        if isinstance(exclude, BlockState):
            exclude = [exclude]

        for z, y, x in np.ndindex(self.shape[::-1]):
            pos = BlockPosition(x, y, z) * self.sign
            state = self[pos]

            if exclude:
                if not ignore_props:
                    if state in exclude:
                        continue
                else:
                    if any(state.id == ex.id for ex in exclude):
                        continue

            if include:
                if not ignore_props:
                    if state not in include:
                        continue
                else:
                    if not any(state.id == s.id for s in include):
                        continue

            yield pos, state

    def to_global(self, local_pos: BlockPosition) -> BlockPosition:
        return self._origin + local_pos

    def to_local(self, global_pos: BlockPosition) -> BlockPosition:
        return global_pos - self._origin

    def to_nbt(self) -> nbtlib.Compound:
        nbt = nbtlib.Compound()

        nbt["Position"] = self._origin.to_nbt()
        nbt["Size"] = self._size.to_nbt()

        pal = [block.to_nbt() for block in self._palette]
        nbt["BlockStatePalette"] = nbtlib.List[nbtlib.Compound](pal)
        nbt["BlockStates"] = self._encode_block_states()

        nbt["Entities"] = self._entities
        nbt["TileEntities"] = self._tile_entities
        nbt["PendingBlockTicks"] = self._block_ticks
        nbt["PendingFluidTicks"] = self._fluid_ticks

        return nbt

    @staticmethod
    def from_nbt(nbt: nbtlib.Compound) -> Region:
        pos = BlockPosition.from_nbt(nbt["Position"])
        size = Size3D.from_nbt(nbt["Size"])

        region = Region(origin=pos, size=size)

        region._palette = [
            BlockState.from_nbt(block) for block in nbt["BlockStatePalette"]]
        region._palette_map = {bl: i for i, bl in enumerate(region._palette)}
        region._blocks = region._decode_block_states(nbt["BlockStates"])

        region._entities = nbt["Entities"]
        region._tile_entities = nbt["TileEntities"]
        region._block_ticks = nbt["PendingBlockTicks"]
        region._fluid_ticks = nbt["PendingFluidTicks"]

        return region
