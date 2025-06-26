from __future__ import annotations

from abc import ABC, abstractmethod
from bitpacking import bitpack, bitunpack
import copy
from itertools import product
import nbtlib
import numpy as np
import twos
from typing import Iterator

from .block_state import BlockState
from .geometry import BlockPosition, Direction, Size3D
from .resource_location import BlockId


AIR = BlockState("air")


class Region:
    def __init__(
        self,
        size: tuple[int, int, int] | Size3D,
        origin: tuple[int, int, int] | BlockPosition = (0, 0, 0),
    ):
        if not isinstance(size, Size3D):
            size = Size3D(*size)
        self._size: Size3D = size

        if not isinstance(origin, BlockPosition):
            origin = BlockPosition(*origin)
        self._origin: BlockPosition = origin

        self._palette: list[BlockState] = [AIR] # TODO: add clear method
        self._palette_map: dict[BlockState, int] = {AIR: 0} # TODO: bind tighter to _palette
        self._blocks = np.zeros(abs(self._size), dtype=int)

        # TODO: Add support for (tile) entities and ticks
        self._entities = nbtlib.List[nbtlib.Compound]()
        self._tile_entities = nbtlib.List[nbtlib.Compound]()
        self._block_ticks = nbtlib.List[nbtlib.Compound]()
        self._fluid_ticks = nbtlib.List[nbtlib.Compound]()

        self._local = LocalRegionView(self)
        self._world = WorldRegionView(self)
        self._numpy = NumpyRegionView(self)
        self._view = self._local

    @property
    def local(self) -> LocalRegionView:
        return self._local

    @property
    def world(self) -> WorldRegionView:
        return self._world

    @property
    def numpy(self) -> NumpyRegionView:
        return self._numpy

    def set_default_view(self, view: _RegionView) -> None:
        self._view = view

    def __contains__(self, item) -> bool:
        return item in self._view

    def __eq__(self, other) -> np.ndarray[bool]:
        return self._view == other

    def __ne__(self, other) -> np.ndarray[bool]:
        return self._view != other

    def __lt__(self, other) -> np.ndarray[bool]:
        return self._view < other

    def __gt__(self, other) -> np.ndarray[bool]:
        return self._view > other

    def __getitem__(self, key):
        return self._view[key]

    def __setitem__(self, key, value) -> None:
        self._view[key] = value

    def __iter__(self) -> tuple[BlockPosition, BlockState]:
        return iter(self._view)


    def clear(self) -> None:
        self._palette: list[BlockState] = [AIR]
        self._palette_map: dict[BlockState, int] = {AIR: 0}
        self._blocks = np.zeros(abs(self._size), dtype=int)

        self._entities.clear()
        self._tile_entities.clear()
        self._block_ticks.clear()
        self._fluid_ticks.clear()

    def copy(
        self,
        origin: tuple[int, int, int] | BlockPosition | None = None,
    ) -> Region:
        if origin is None:
            origin = self._origin

        reg = Region(size=self._size, origin=origin)
        reg._blocks = copy.deepcopy(self._blocks)
        reg._palette = copy.deepcopy(self._palette)
        reg._palette_map = copy.deepcopy((self._palette_map))

        reg._entities = copy.deepcopy(self._entities)
        reg._tile_entities = copy.deepcopy(self._tile_entities)
        reg._block_ticks = copy.deepcopy(self._block_ticks)
        reg._fluid_ticks = copy.deepcopy(self._fluid_ticks)
        reg._view = type(self._view)(reg)

        return reg

    def flip(self, axis: int | tuple[int] | None = None) -> None:
        # TODO
        # * flip size and possibly change origin?
        # * handle entities
        # * change BlockState properties accordingly
        self._blocks = np.flip(self._blocks, axis=axis)

    def compact_palette(self) -> None:
        # TODO: determine all appropriate places to call this method
        idx = np.unique(self._blocks)
        if 0 not in idx:
            # always include minecraft:air as the first entry in the palette
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

    # block state en- / decoding (NBT)
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

    @property
    def size(self) -> Size3D:
        return self._size

    @property
    def width(self) -> int:
        return self.size.width

    @property
    def height(self) -> int:
        return self.size.height

    @property
    def length(self) -> int:
        return self.size.length

    @property
    def volume(self) -> int:
        return self.size.volume

    @property
    def origin(self) -> BlockPosition:
        return self._origin

    @origin.setter
    def origin(self, value: tuple[int, int, int] | BlockPosition) -> None:
        if not isinstance(value, BlockPosition):
            value = BlockPosition(*value)
        self._origin = value

    def count(self, block: BlockState | BlockId) -> int:
        return np.sum(self == block).item()

    @property
    def lower(self) -> BlockPosition:
        return self.local.lower
        # return self._view.lower

    @property
    def upper(self) -> BlockPosition:
        return self.local.upper
        # return self._view.upper

    @property
    def bounds(self) -> tuple[BlockPosition, BlockPosition]:
        return self.local.bounds
        # return self._view.bounds

    def items(self) -> Iterator[tuple[BlockPosition, BlockState]]:
        return self._view.items()

    def positions(self) -> Iterator[BlockPosition]:
        return self._view.positions()

    def blocks(self) -> Iterator[BlockState]:
        return self._view.blocks()

    # masking relative to BlockState / BlockId
    def relative_to(
        self,
        block: BlockState | BlockId,
        direction: BlockPosition,
    ) -> np.ndarray[bool]:
        return self._view.relative_to(block, direction)

    def above_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self._view.above_of(block)

    def below_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self._view.below_of(block)

    def north_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self._view.north_of(block)

    def south_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self._view.south_of(block)

    def west_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self._view.west_of(block)

    def east_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self._view.east_of(block)

    # block position transformations
    def world_to_local(self, world: BlockPosition) -> BlockPosition:
        return world - self._origin

    def local_to_world(self, local: BlockPosition) -> BlockPosition:
        return self._origin + local

    def local_to_numpy(self, local: BlockPosition) -> BlockPosition:
        return BlockPosition(*self.local.position_to_index(local))

    def numpy_to_local(self, index: BlockPosition) -> BlockPosition:
        return self.local.index_to_position(tuple(index))

    def world_to_numpy(self, world: BlockPosition) -> BlockPosition:
        return BlockPosition(*self.world.position_to_index(world))

    def numpy_to_world(self, index: BlockPosition) -> BlockPosition:
        return self.world.index_to_position(tuple(index))

    # NBT conversion
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

    @classmethod
    def from_nbt(cls, nbt: nbtlib.Compound) -> Region:
        pos = BlockPosition.from_nbt(nbt["Position"])
        size = Size3D.from_nbt(nbt["Size"])

        region = cls(origin=pos, size=size)

        region._palette = [
            BlockState.from_nbt(block) for block in nbt["BlockStatePalette"]]
        region._palette_map = {bl: i for i, bl in enumerate(region._palette)}
        region._blocks = region._decode_block_states(nbt["BlockStates"])

        region._entities = nbt["Entities"]
        region._tile_entities = nbt["TileEntities"]
        region._block_ticks = nbt["PendingBlockTicks"]
        region._fluid_ticks = nbt["PendingFluidTicks"]

        return region


class _RegionView(ABC):

    def __init__(self, region: Region) -> None:
        self.region = region
        # TODO: add support for (tile) entities and ticks

    @property
    def _blocks(self) -> np.ndarray[int]:
        return self.region._blocks

    @property
    def _palette(self) -> list[BlockState]:
        return self.region._palette

    @property
    def _palette_map(self) -> dict[BlockState, int]:
        return self.region._palette_map

    @abstractmethod
    def position_to_index(self, pos: BlockPosition) -> tuple[int, int, int]:
        ...

    @abstractmethod
    def index_to_position(self, index: tuple[int, int, int]) -> BlockPosition:
        ...

    @abstractmethod
    def _align_array(self, arr: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def _transform_index(self, index):
        ...

    def _block_mask(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        if not isinstance(block, (BlockState, BlockId)):
            raise TypeError(f"'block' needs to be BlockState or BlockId")

        palette = np.array(self._palette, dtype=object)

        # matches = np.array([block == state for state in palette])
        # return matches[self._blocks]
        if isinstance(block, BlockState):
            matches = np.array([state == block for state in palette])
        else:
            matches = np.array([state.id == block for state in palette])

        return matches[self._blocks]

    def __getitem__(self, key):
        if isinstance(key, BlockPosition):
            # return self.at(key) # TODO
            key = tuple(key)
        index = self._transform_index(key)

        indices = self._blocks[index]
        if np.isscalar(indices):
            return self._palette[indices]
        else:
            return np.array(self._palette, dtype=object)[indices]

    def __setitem__(self, key, value):
        if isinstance(key, BlockPosition):
            # return self.set_at(key, value) # TODO
            key = tuple(key)
        index = self._transform_index(key)

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
                # TODO: allow casting
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

    def __contains__(self, item) -> bool:
        if isinstance(item, BlockPosition):
            return all(self.lower <= item) and all(item <= self.upper)

        elif isinstance(item, BlockState):
            index = self._palette_map.get(item)
            if index is None:
                return False
            return index in self._blocks
            # return np.any(self._blocks == index).item()

        elif isinstance(item, BlockId):
            return any(
                # bs.id == item and np.any(self._blocks == idx)
                bs.id == item and idx in self._blocks
                for bs, idx in self._palette_map.items())

        else:
            return False

    def __eq__(self, other) -> np.ndarray[bool]:
        if not isinstance(other, (BlockState, BlockId)):
            return NotImplemented
        mask = self._block_mask(other)
        return self._align_array(mask)

    def __ne__(self, other) -> np.ndarray[bool]:
        if not  isinstance(other, (BlockState, BlockId)):
            return NotImplemented
        return np.invert(self.__eq__(other))

    def __lt__(self, other) -> np.ndarray[bool]:
        if not  isinstance(other, (BlockState, BlockId)):
            return NotImplemented
        return self.below_of(other)

    def __gt__(self, other) -> np.ndarray[bool]:
        if not  isinstance(other, (BlockState, BlockId)):
            return NotImplemented
        return self.above_of(other)

    def __iter__(self) -> Iterator[tuple[BlockPosition, BlockState]]:
        return self.items()

    def items(self) -> Iterator[tuple[BlockPosition, BlockState]]:
        for pos, block in zip(self.positions(), self.blocks()):
            yield pos, block

    def positions(self) -> Iterator[BlockPosition]:
        ranges = [
            range(start, stop, step)
            for start, stop, step
            in zip(self.origin, self.origin + self.size, self.size.sign)
        ]
        for z, y, x in product(*reversed(ranges)):
            yield BlockPosition(x, y, z)

    def blocks(self) -> Iterator[BlockState]:
        indices = self._align_array(self._blocks).transpose(2, 1, 0).ravel()
        palette = np.array(self._palette, dtype=object)
        for block in palette[indices]:
            yield block

    def _move_mask(
        self,
        mask: np.ndarray[bool],
        direction: BlockPosition,
    ) -> np.ndarray[bool]:
        result = np.zeros_like(mask, dtype=bool)

        slices_src = [slice(None)] * self._blocks.ndim
        slices_dst = [slice(None)] * self._blocks.ndim

        for axis, dim in enumerate(direction):
            if dim == 0:
                continue
            elif dim > 0:
                slices_src[axis] = slice(0, -dim)
                slices_dst[axis] = slice(dim, None)
            else:
                slices_src[axis] = slice(-dim, None)
                slices_dst[axis] = slice(0, dim)

        result[tuple(slices_dst)] = mask[tuple(slices_src)]
        return result

    def relative_to(
        self,
        block: BlockState | BlockId,
        direction: BlockPosition,
    ) -> np.ndarray[bool]:
        mask = self._block_mask(block)
        mask = self._move_mask(mask, direction)
        return self._align_array(mask)

    def above_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self.relative_to(block, Direction.UP)

    def below_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self.relative_to(block, Direction.DOWN)

    def north_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self.relative_to(block, Direction.NORTH)

    def south_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self.relative_to(block, Direction.SOUTH)

    def west_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self.relative_to(block, Direction.WEST)

    def east_of(self, block: BlockState | BlockId) -> np.ndarray[bool]:
        return self.relative_to(block, Direction.EAST)

    property
    @abstractmethod
    def origin(self) -> BlockPosition:
        ...

    @property
    @abstractmethod
    def size(self) -> Size3D:
        ...

    @property
    def limit(self) -> BlockPosition:
        return self.origin + self.size.limit

    @property
    def lower(self) -> BlockPosition:
        return BlockPosition(*np.min((self.origin, self.limit), axis=0))

    @property
    def upper(self) -> BlockPosition:
        return BlockPosition(*np.max((self.origin, self.limit), axis=0))

    @property
    def bounds(self) -> tuple[BlockPosition, BlockPosition]:
        return self.lower, self.upper


class NumpyRegionView(_RegionView):

    @property
    def origin(self) -> BlockPosition:
        return BlockPosition(0, 0, 0)

    @property
    def size(self) -> Size3D:
        return abs(self.region._size)

    def position_to_index(self, pos: BlockPosition) -> tuple[int, int, int]:
        return tuple(pos)

    def index_to_position(self, index: tuple[int, int, int]) -> BlockPosition:
        return BlockPosition(*index)

    def _align_array(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def _transform_index(self, index):
        return index


class _OrientedView(_RegionView):

    @property
    def size(self) -> Size3D:
        return self.region._size

    @property
    def negative_axes(self) -> tuple[int,...]:
        return tuple(np.argwhere(self.size < 0).flatten().tolist())

    def position_to_index(self, pos: BlockPosition) -> tuple[int, int, int]:
        return pos - self.lower

    def index_to_position(self, index: tuple[int, int, int]) -> BlockPosition:
        return self.lower + index

    def _align_array(self, arr: np.ndarray) -> np.ndarray:
        return np.flip(arr, axis=self.negative_axes)

    def _transform_index(self, key):
        if isinstance(key, (int, np.integer, slice, type(Ellipsis))):
            key = (key,)

        if isinstance(key, tuple):
            key = list(key)
            for i, k in enumerate(key):
                offset = self.lower[i]
                if isinstance(k, (int, np.integer)):
                    key[i] = k - offset
                elif isinstance(k, slice):
                    start = k.start - offset if k.start is not None else None
                    stop = k.stop - offset if k.stop is not None else None
                    key[i] = slice(start, stop, k.step)
                else:
                    # Ellipsis
                    key[i] = k
            return tuple(key)

        elif isinstance(key, np.ndarray) and key.dtype == bool:
            # boolean indexing
            key = self._align_array(key)
            if key.shape != self._blocks.shape:
                raise IndexError("Boolean index must match region shape.")
            return key

        else:
            return key


class LocalRegionView(_OrientedView):

    @property
    def origin(self) -> BlockPosition:
        return BlockPosition(0, 0, 0)


class WorldRegionView(_OrientedView):

    @property
    def origin(self) -> BlockPosition:
        return self.region._origin
