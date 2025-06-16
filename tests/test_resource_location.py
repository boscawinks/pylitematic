from nbtlib import String

from pylitematic.resource_location import ResourceLocation


def test_resource_location():
    # TODO
    # * test sorting of resource locations
    # * test invalid strings

    def check_location(
        location: ResourceLocation,
        namespace: str,
        path: str,
        string: str,
        nbt: String,
    ) -> None:
        assert location.namespace == namespace
        assert location.path == path
        assert str(location) == string
        assert location.to_string() == string
        assert location.to_nbt() == nbt

    proto_locations = [
        ("minecraft", "air", "minecraft:air", "minecraft:air"),
        ("minecraft", "stone", "stone", "minecraft:stone"),
        ("minecraft", "dirt", ":dirt", "minecraft:dirt"),
        ("a_mod", "maple_log", "a_mod:maple_log", "a_mod:maple_log"),
        ("a_mod", "dir/cfg.json", "a_mod:dir/cfg.json", "a_mod:dir/cfg.json"),
    ]

    for namespace, path, string, loc_str in proto_locations:
        loc = ResourceLocation(namespace=namespace, path=path)
        check_location(loc, namespace, path, loc_str, String(loc_str))
        loc = ResourceLocation.from_string(string=string)
        check_location(loc, namespace, path, loc_str, String(loc_str))
        loc = ResourceLocation.from_nbt(nbt=String(string))
        check_location(loc, namespace, path, loc_str, String(loc_str))
