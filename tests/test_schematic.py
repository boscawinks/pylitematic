from pylitematic import BlockPosition, BlockState, Region, Schematic, Size3D


def test_schematic(tmp_path):
    air = BlockState("air")
    stone = BlockState("stone")
    dirt = BlockState("dirt")
    grass = BlockState("grass_block")
    cobble = BlockState("mossy_cobblestone")
    snow = BlockState("snow_block")
    pumpkin = BlockState("carved_pumpkin", facing="west")

    ground = Region(size=Size3D(16, 9, 16))
    ground[:,:5,:] = stone
    ground[:,5:8,:] = dirt
    ground[:,8:,:] = grass

    boulder = Region(
        size=(4, 4, 4), origin=ground.origin+[6, ground.height, 6])
    for pos, block in boulder.blocks():
        if block == air:
            boulder[pos] = cobble
    # ^ since region is empty, this is equivalent to boulder[:] = cobble

    snow_man = Region(
        size=(1, 3, 1), origin=boulder.origin+[1, boulder.height, 1])
    snow_man[:] = snow
    snow_man[BlockPosition(0, 2, 0)] = pumpkin

    schem = Schematic(
        name="scene", author="Boscawinks", description="A simple scene")
    schem.add_region("ground", ground)
    schem.add_region("boulder", boulder)
    schem.add_region("snow_man", snow_man)

    save_path = tmp_path / f"{schem.name}.litematic"
    schem.save(save_path)

    copy = schem.load(save_path)
    assert copy.to_nbt() == schem.to_nbt()
