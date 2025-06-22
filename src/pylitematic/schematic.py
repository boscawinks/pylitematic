from __future__ import annotations

from datetime import datetime
import nbtlib
import numpy as np
import pathlib
import time
import twos
from typing import Iterator

from pylitematic.geometry import BlockPosition, Size3D
from pylitematic.region import Region


DEFAULT_VERSION_MAJOR: int = 7
DEFAULT_VERSION_MINOR: int = 1
DEFAULT_MC_VERSION: int = 4325

PREVIEW_CHANNELS: int = 4
PREVIEW_BIT_DEPTH: int = 8


def decode_image_data(data: nbtlib.IntArray) -> np.ndarray[int]:
    bit_mask = (1 << PREVIEW_BIT_DEPTH) - 1

    data = np.vectorize(twos.to_unsigned)(
        data.unpack(), PREVIEW_CHANNELS * PREVIEW_BIT_DEPTH)
    size = int(np.sqrt(len(data))) # TODO: Handle non-squares
    shape = (size, size, PREVIEW_CHANNELS)
    pixels = np.zeros(data.shape + (PREVIEW_CHANNELS,), dtype=int)
    for i in range(PREVIEW_CHANNELS):
        pixels[..., i] = (data >> (i * PREVIEW_BIT_DEPTH) & bit_mask)
    pixels = np.asarray(pixels, dtype=np.uint32).reshape(shape)
    return pixels[:, :, [2, 1, 0, 3]] # BGRA -> RGBA


def encode_image_data(img: np.ndarray[int]) -> nbtlib.IntArray:
    bit_mask = (1 << PREVIEW_BIT_DEPTH) - 1

    # TODO: Handle images w/o alpha channel
    img = img[:, :, [2, 1, 0, 3]] # RGBA -> BGRA
    img = img.reshape(-1, PREVIEW_CHANNELS)
    data = np.zeros(len(img), dtype=int)
    for i in range(PREVIEW_CHANNELS):
        data |= ((img[..., i] & bit_mask )<< (i * PREVIEW_BIT_DEPTH))
    data = np.vectorize(twos.to_signed)(
        data, PREVIEW_CHANNELS * PREVIEW_BIT_DEPTH)
    return nbtlib.IntArray(data)


class Schematic:
    def __init__(
        self,
        name: str,
        author: str = "",
        description: str = "",
        regions: dict[str, Region] = {},
        preview: np.ndarray[int] | None = None,
        version_major: int = DEFAULT_VERSION_MAJOR,
        version_minor: int | None = DEFAULT_VERSION_MINOR,
        mc_version: int = DEFAULT_MC_VERSION,
    ) -> None:
        self.name = name
        self.description = description
        self.author = author
        self._preview = preview

        self._regions: dict[str, Region] = regions

        self._created_at = round(time.time() * 1000)
        self._modified_at = self._created_at

        self.version_major = version_major
        self.version_minor = version_minor
        self.mc_version = mc_version

        self.modified: bool = True

    def __getitem__(self, key):
        return self._regions[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Region):
            raise TypeError(
                f"Can only add Region objects, got {type(value).__name__}")
        self._regions[key] = value

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
    def size(self) -> Size3D:
        lower, upper = self.bounds
        return Size3D(*abs(upper - lower)) + 1

    @property
    def bounds(self) -> tuple[BlockPosition, BlockPosition]:
        lowers = []
        uppers = []
        for reg in self._regions.values():
            lower, upper = reg.global_bounds
            lowers.append(lower)
            uppers.append(upper)
        return (
            BlockPosition(*np.min(lowers, axis=0)),
            BlockPosition(*np.max(uppers, axis=0)),
        )

    @property
    def volume(self) -> int:
        return sum(reg.volume for reg in self._regions.values())

    @property
    def blocks(self) -> int:
        return sum(reg.block_count for reg in self._regions.values())

    @property
    def region_count(self) -> int:
        return len(self._regions)

    @property
    def version(self) -> str:
        if self.version_minor is None:
            return str(self.version_major)
        return f"{self.version_major}.{self.version_minor}"

    @property
    def preview(self) -> np.ndarray[int] | None:
        return self._preview

    def regions(self) -> Iterator[tuple[str, Region]]:
        return self._regions.items()

    def add_region(self, name: str, region: Region) -> None:
        self._regions[name] = region

    def remove_region(self, name: str) -> Region:
        return self._regions.pop(name)

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(int(self._created_at / 1000))

    @property
    def modified_at(self) -> datetime:
        return datetime.fromtimestamp(int(self._modified_at / 1000))

    def save(self, path: pathlib.Path | str) -> None:
        file = nbtlib.File(self.to_nbt())
        file.save(path, gzipped=True, byteorder="big")

    @staticmethod
    def load(path: pathlib.Path | str) -> Schematic:
        if isinstance(path, str):
            path = pathlib.Path(path)
        nbt = nbtlib.File.load(path.expanduser(), True)
        return Schematic.from_nbt(nbt)

    def to_nbt(self) -> nbtlib.Compound:
        # self._update()
        nbt = nbtlib.Compound()

        # meta data
        meta = nbtlib.Compound()
        meta["Name"] = nbtlib.String(self.name)
        meta["Author"] = nbtlib.String(self.author)
        meta["Description"] = nbtlib.String(self.description)

        meta["TimeCreated"] = nbtlib.Long(self._created_at)
        meta["TimeModified"] = nbtlib.Long(self._modified_at)

        meta["RegionCount"] = nbtlib.Int(self.region_count)
        if self._preview is not None:
            meta["PreviewImageData"] = encode_image_data(self._preview)

        meta["EnclosingSize"] = self.size.to_nbt()
        meta["TotalVolume"] = nbtlib.Long(self.volume)
        meta["TotalBlocks"] = nbtlib.Long(self.blocks)
        nbt["Metadata"] = meta

        # regions
        regions = nbtlib.Compound()
        for name, region in self.regions():
            region.compact_palette()
            regions[name] = region.to_nbt()
        nbt["Regions"] = regions

        # versions
        nbt["Version"] = nbtlib.Int(self.version_major)
        if self.version_minor is not None:
            nbt["SubVersion"] = nbtlib.Int(self.version_minor)
        nbt["MinecraftDataVersion"] = nbtlib.Int(self.mc_version)

        return nbt

    @staticmethod
    def from_nbt(nbt: nbtlib.Compound) -> Schematic:
        # meta data
        try:
            meta = nbt["Metadata"]
        except KeyError as exc:
            raise KeyError(
                "Can't load from NBT without 'Metadata' entry") from exc

        name = meta["Name"].unpack()
        author = meta["Author"].unpack()
        try:
            desc = meta["Description"].unpack()
        except KeyError:
            desc = ""

        preview = meta.get("PreviewImageData")
        if preview is not None:
            preview = decode_image_data(preview)

        created_at = meta["TimeCreated"].unpack()
        modified_at = meta["TimeModified"].unpack()

        # regions
        try:
            regions = nbt["Regions"]
        except KeyError as exc:
            raise KeyError(
                "Can't load from NBT without 'Regions' entry") from exc
        regions = {name: Region.from_nbt(nbt) for name, nbt in regions.items()}

        # version(s)
        try:
            major = nbt["Version"].unpack()
        except KeyError as exc:
            raise KeyError(
                "Can't load from NBT without 'Version' entry") from exc

        try:
            minor = nbt["SubVersion"].unpack()
        except KeyError:
            minor = None

        mc_version = nbt.get("MinecraftDataVersion")

        schem = Schematic(
            name=name,
            author=author,
            description=desc,
            regions=regions,
            preview=preview,
            version_major=major,
            version_minor=minor,
            mc_version=mc_version,
        )

        schem._created_at = created_at
        schem._modified_at = modified_at

        return schem
