from __future__ import annotations

import copy
import nbtlib
import numpy as np
from typing import Iterator

from .block_state import AIR, BlockId, BlockState


class BlockPalette:
    def __init__(self):
        self._states: list[BlockState] = []
        self._map: dict[BlockState, int] = {}
        self.add_state(AIR)

    def __len__(self):
        return len(self._states)

    def __str__(self) -> str:
        return str({str(s): i for s, i in self._map.items()})

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._map})"

    def __contains__(self, key) -> bool:
        # TODO: add BlockId comparison?
        if not isinstance(key, BlockState):
            return NotImplemented
        return key in self._map

    def __eq__(self, other) -> bool | np.ndarray[bool]:
        if not isinstance(other, (BlockState, BlockId)):
            return NotImplemented
        return np.array(self) == other

    def __iter__(self) -> Iterator[tuple[BlockState, int]]:
        return self.items()

    def __array__(self, dtype: type | None = None, copy: bool = True):
        arr = np.array(self._states, dtype=object)
        return arr.copy() if copy else arr

    def states(self) -> Iterator[BlockState]:
        for state in self._map.keys():
            yield state

    def indices(self) -> Iterator[int]:
        for index in self._map.values():
            yield index

    def items(self) -> Iterator[tuple[BlockState, int]]:
        for state, index in self._map.items():
            yield state, index

    @property
    def bits_per_state(self) -> int:
        return max(2, (len(self) - 1).bit_length())

    def copy(self) -> BlockPalette:
        pal = BlockPalette()
        pal._states = copy.deepcopy(self._states)
        pal._map = copy.deepcopy(self._map)
        return pal

    def clear(self) -> None:
        self._states.clear()
        self._map.clear()
        self.add_state(AIR)

    def add_state(self, state: BlockState) -> int:
        if state in self:
            return self._map[state]
        else:
            index = len(self)
            self._states.append(state)
            self._map[state] = index
            return index

    def get_state(
        self,
        index: int | np.ndarray[int],
    ) -> BlockState | np.ndarray[BlockState]:
        return np.array(self._states, dtype=object)[np.array(index, dtype=int)]

    def get_index(
        self,
        state: BlockState | np.ndarray[BlockState],
        add_missing: bool = False,
    ) -> int | np.ndarray[int]:
        state = np.asarray(state, dtype=object)
        unique_states, xdi = np.unique(state, return_inverse=True)
        idx = []
        for block in unique_states:
            if block not in self and not add_missing:
                raise KeyError(f"BlockState '{block!s}' not found in palette.")
            idx.append(self.add_state(block))

        index = np.array(idx, dtype=int)[xdi].reshape(state.shape)
        return index.item() if np.isscalar(index) else index

    def reduce(self, indices: np.ndarray[int]) -> None:
        if not (isinstance(indices, np.ndarray) and indices.dtype == int):
            raise TypeError("'indices' has to be a numpy array of integers")

        unique_idx = np.unique(indices)
        if 0 not in unique_idx:
            # always include minecraft:air as the first entry in the palette
            unique_idx = np.insert(unique_idx, 0, 0)
        self._states = np.array(self._states, dtype=object)[unique_idx].tolist()
        self._map = {state: i for i, state in enumerate(self._states)}

        old_new_map = {old: new for new, old in enumerate(unique_idx)}
        lookup = np.full(max(old_new_map) + 1, -1, dtype=int)
        for old, new in old_new_map.items():
            lookup[old] = new
        return lookup[indices]

    def to_nbt(self) -> nbtlib.List[nbtlib.Compound]:
        pal = [state.to_nbt() for state in self._states]
        return nbtlib.List[nbtlib.Compound](pal)

    @classmethod
    def from_nbt(cls, nbt: nbtlib.List[nbtlib.Compound]) -> BlockPalette:
        states = [BlockState.from_nbt(block) for block in nbt]
        pal = cls()
        pal._states = states
        pal._map = {state: i for i, state in enumerate(states)}
        return pal
