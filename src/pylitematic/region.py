from __future__ import annotations

from bitpacking import bitpack, bitunpack
import nbtlib
import numpy as np
import twos

from .block_state import BlockState
from .geometry import BlockPosition, Size3D


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
        self._palette_map: dict[BlockState, int] = {AIR: 0} # FIXME
        self._blocks = np.zeros(abs(self._size))
        self._entities: list[nbtlib.Compound] = []
        self._tile_entities: list[nbtlib.Compound] = []
        self._block_ticks: list[nbtlib.Compound] = []
        self._fluid_ticks: list[nbtlib.Compound] = []

    def __getitem__(self, key):
        if isinstance(key, BlockPosition):
            index = self._blocks[key.x, key.y, key.z]
            return self._palette[index]

        indices = self._blocks[key]
        if np.isscalar(indices):
            return self._palette[indices]

        return np.array(self._palette, dtype=object)[indices]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, BlockPosition):
            key = key.to_tuple()

        if isinstance(value, list):
            value = np.array(value, dtype=object)

        if isinstance(value, BlockState):
            # assign single BlockState to slice
            if value not in self._palette_map:
                self._palette_map[value] = len(self._palette)
                self._palette.append(value)
            index = self._palette_map[value]
            self._blocks[key] = index

        elif isinstance(value, np.ndarray):
            if value.shape != self._blocks[key].shape:
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
            self._blocks[key] = index_array
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

    @property
    def size(self) -> Size3D:
        return self._size

    @property
    def origin(self) -> BlockPosition:
        return self._origin

    @property
    def end(self) -> BlockPosition:
        return self._origin + np.where(
            self._size > 0, self._size - 1, self._size + 1)

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
    def blocks(self) -> int: # TODO: rename
        return np.count_nonzero(self._blocks)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._blocks.shape

    @property
    def lower(self) -> BlockPosition:
        return BlockPosition(*np.min((self.origin, self.end), axis=0))

    @property
    def upper(self) -> BlockPosition:
        return BlockPosition(*np.max((self.origin, self.end), axis=0))

    @property
    def bounds(self) -> tuple[BlockPosition, BlockPosition]:
        return self.lower, self.upper

    # def count(self, block: BlockState, ignore_props: bool = False) -> int:
    #     ...

    def global_position(self, pos: BlockPosition) -> BlockPosition:
        return self._origin + pos

    def local_position(self, pos: BlockPosition) -> BlockPosition:
        return pos - self._origin

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
