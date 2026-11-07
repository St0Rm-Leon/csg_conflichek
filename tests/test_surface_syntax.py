import pytest

from csg_conflichek import macro as mb
from csg_conflichek import shapes as sp
from csg_conflichek._geom_syntax_core import SurfaceSyntaxError
from csg_conflichek.phits_geom_syntax import parse_phits_surface, parse_phits_surfaces


@pytest.mark.parametrize(
    "card, shape_type",
    [
        ("1 P 1 0 0 0", sp.Plane),
        ("1 PX 0", sp.XPlane),
        ("1 PY 0", sp.YPlane),
        ("1 PZ 0", sp.ZPlane),
        ("1 SO 1", sp.OSphere),
        ("1 S 0 0 0 1", sp.Sphere),
        ("1 SX 0 1", sp.XSphere),
        ("1 SY 0 1", sp.YSphere),
        ("1 SZ 0 1", sp.ZSphere),
        ("1 C/X 0 0 1", sp.XCylinder),
        ("1 C/Y 0 0 1", sp.YCylinder),
        ("1 C/Z 0 0 1", sp.ZCylinder),
        ("1 CX 1", sp.CX),
        ("1 CY 1", sp.CY),
        ("1 CZ 1", sp.CZ),
        ("1 K/X 0 0 0 1 1", sp.XCone),
        ("1 KX 0 1", sp.KX),
        ("1 KY 0 1", sp.KY),
        ("1 KZ 0 1", sp.KZ),
        ("1 SQ 1 1 1 0 0 0 -1 0 0 0", sp.SQ),
        ("1 GQ 1 1 1 0 0 0 0 0 0 -1", sp.General),
        ("1 TX 0 0 0 2 1 1", sp.XTorus),
        ("1 TY 0 0 0 2 1 1", sp.YTorus),
        ("1 TZ 0 0 0 2 1 1", sp.ZTorus),
        ("1 RPP -1 1 -2 2 -3 3", mb.RPP),
        ("1 RCC 0 0 0 0 0 2 1", mb.RCC),
        ("1 BOX 0 0 0 1 0 0 0 1 0 0 0 1", mb.BOX),
        ("1 HEX 0 0 0 0 0 1 1 0 0 0.5 0.8660254 0 -0.5 0.8660254 0", mb.HEX),
        ("1 RHP 0 0 0 0 0 1 1 0 0 0.5 0.8660254 0 -0.5 0.8660254 0", mb.HEX),
        ("1 REC 0 0 0 0 0 2 1 0 0 0.5", mb.REC),
        ("1 TRC 0 0 0 0 0 10 5 2", mb.TRC),
        ("1 ELL 3 3 0 -3 -3 0 9", mb.ELL),
        ("1 WED 0 0 0 10 0 0 0 10 0 0 0 5", mb.WED),
        (
            "1 ARB -5 -10 -5 -5 -10 5 5 -10 -5 5 -10 5 "
            "0 12 0 0 0 0 0 0 0 0 0 0 "
            "1234 1250 1350 2450 3450 0",
            mb.ARB,
        ),
        ("1 SPH 0 0 0 1", sp.Sphere),
    ],
)
def test_parse_surface_types(card, shape_type):
    parsed = parse_phits_surface(card)

    assert parsed.surface_id == 1
    assert isinstance(parsed.shape, shape_type)


def test_parse_surface_known_point():
    parsed = parse_phits_surface("1 SO 2")

    assert parsed.shape.is_positive(0.0, 0.0, 0.0) is False
    assert parsed.shape.is_positive(3.0, 0.0, 0.0) is True


def test_parse_surface_optional_transform():
    assert parse_phits_surface("1 SO 2").transform is None
    assert parse_phits_surface("1 0 SO 2").transform == 0
    assert parse_phits_surface("1 7 SO 2").transform == 7


def test_phits_surface_parser():
    parsed = parse_phits_surface("1 0 so 2")

    assert parsed.surface_id == 1
    assert parsed.transform == 0
    assert isinstance(parsed.shape, sp.OSphere)


def test_parse_surface_block_and_duplicate():
    surfaces = parse_phits_surfaces(["1 SO 1", "2 PX 0", "$ comment"])

    assert set(surfaces) == {1, 2}

    with pytest.raises(SurfaceSyntaxError, match="duplicate"):
        parse_phits_surfaces(["1 SO 1", "1 SO 2"])


@pytest.mark.parametrize(
    "card",
    [
        "0 SO 1",
        "1 TRC 0 0 0 1 2 3 0 0 1",
        "1 SO",
        "1 SO 1 2",
        "1 P 1 0 0",
        "1 P 1 0 0 x",
        "1 PX 0 1",
    ],
)
def test_invalid_surface_cards(card):
    with pytest.raises(SurfaceSyntaxError):
        parse_phits_surface(card)
