import numpy as np
import pytest

from csg_conflichek import shapes as sp
from csg_conflichek._geom_syntax_core import (
    CellSyntaxError,
    InputSyntaxError,
    SurfaceSyntaxError,
    TransformedShape,
)
from csg_conflichek.phits_geom_syntax import (
    load_phits_text,
    parse_phits_transform_line,
)
from csg_conflichek.transforms import Transform, rotation_z


def test_identity_and_inverse():
    transform = Transform(rotation_z(np.pi / 2), np.array([1.0, 2.0, 3.0]))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    global_points = transform.local_to_global(points)
    local_points = transform.global_to_local(global_points)

    np.testing.assert_allclose(local_points, points, atol=1e-12)


def test_translation_and_inverse():
    transform = Transform(np.eye(3), np.array([10.0, -2.0, 3.0]))

    np.testing.assert_allclose(
        transform.local_to_global(np.array([1.0, 0.0, 0.0])),
        [11.0, -2.0, 3.0],
    )
    np.testing.assert_allclose(
        transform.global_to_local(np.array([11.0, -2.0, 3.0])),
        [1.0, 0.0, 0.0],
    )


def test_compose():
    first = Transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    second = Transform(rotation_z(np.pi / 2), np.zeros(3))
    combined = first.compose(second)

    point = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        combined.local_to_global(point),
        first.local_to_global(second.local_to_global(point)),
    )


def test_phits_m1_rotation():
    b = [0, 1, 0, -1, 0, 0, 0, 0, 1]
    transform = Transform.from_phits(number=1, values=[0, 0, 0, *b], mode=1)

    np.testing.assert_allclose(
        transform.local_to_global(np.array([1.0, 0.0, 0.0])),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_phits_m1_and_m2_translation_order():
    m1 = Transform.from_phits(
        number=1,
        values=[10, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        mode=1,
    )
    m_neg1 = Transform.from_phits(
        number=1,
        values=[10, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        mode=-1,
    )

    np.testing.assert_allclose(
        m1.local_to_global(np.array([1.0, 0.0, 0.0])),
        [11.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        m_neg1.local_to_global(np.array([11.0, 0.0, 0.0])),
        [1.0, 0.0, 0.0],
    )


def test_phits_m2_degrees_and_radians():
    values = [0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0]
    degrees = Transform.from_phits(2, values, 2, degrees=True)
    radians = Transform.from_phits(
        2,
        [0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0],
        2,
    )

    np.testing.assert_allclose(degrees.rotation, radians.rotation)


def test_transformed_shape_rotates_a_plane():
    shape = TransformedShape(
        sp.Plane(1, 0, 0, 0),
        Transform(rotation_z(np.pi / 2), np.zeros(3)),
    )

    assert bool(shape.is_positive(0.0, 1.0, 0.0)) is True
    assert bool(shape.is_positive(1.0, -1.0, 0.0)) is False


def test_phits_m3_center():
    values = [0, 0, 0, 10, 0, 0, 3, 90, 0, 0, 0, 0]
    transform = Transform.from_phits(3, values, 3, degrees=True)

    np.testing.assert_allclose(
        transform.local_to_global(np.array([11.0, 0.0, 0.0])),
        [10.0, 1.0, 0.0],
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", [0, 4, -4])
def test_invalid_phits_mode(mode):
    with pytest.raises(ValueError):
        Transform.from_phits(1, [0] * 12, mode)


def test_parse_phits_transform_line():
    transform = parse_phits_transform_line("tr1 10 0 0 1 0 0 0 1 0 0 0 1 1")

    np.testing.assert_allclose(transform.translation, [10, 0, 0])


def test_parse_phits_degree_transform_line():
    transform = parse_phits_transform_line("*tr1 0 0 0 90 0 0 0 0 0 0 0 0 2")

    np.testing.assert_allclose(
        transform.rotation,
        rotation_z(np.pi / 2),
        atol=1e-12,
    )


PHITS_TRANSFORMED_SURFACE = """title
[Transform]
tr1 10 0 0 1 0 0 0 1 0 0 0 1 1
[Cell]
1 0 -1
[Surface]
1 1 SO 2
"""


PHITS_TRANSFORMED_CELL = """title
[Transform]
tr1 10 0 0 1 0 0 0 1 0 0 0 1 1
[Cell]
1 0 -1 trcl=1
[Surface]
1 SO 2
"""


def test_phits_surface_transform_loading():
    model = load_phits_text(PHITS_TRANSFORMED_SURFACE)

    assert 1 in model.transforms
    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == -1


def test_phits_cell_transform_loading():
    model = load_phits_text(PHITS_TRANSFORMED_CELL)

    assert model.cells[0].transform_id == 1
    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == -1


def test_inline_trcl_translation():
    text = """title
[Cell]
1 0 -1 trcl=(0 0 20)
[Surface]
1 SO 2
"""
    model = load_phits_text(text)

    assert model.transforms == {}
    assert model.cells[0].inline_transform is not None
    assert model.geometry.find_cell(0.0, 0.0, 20.0) == 0
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == -1


def test_inline_trcl_twelve_values_defaults_to_mode_one():
    text = """title
[Cell]
1 0 -1 trcl=(10 0 0 1 0 0 0 1 0 0 0 1)
[Surface]
1 SO 2
"""
    model = load_phits_text(text)

    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0


def test_inline_trcl_thirteen_values_and_degree_marker():
    text = """title
[Cell]
1 0 -1 *trcl=(0 0 0 90 0 0 0 0 0 0 0 0 2)
[Surface]
1 P 1 0 0 0
"""
    model = load_phits_text(text)

    assert model.geometry.find_cell(0.0, -1.0, 0.0) == 0
    assert model.geometry.find_cell(0.0, 1.0, 0.0) == -1


def test_inline_trcl_with_other_options():
    text = """title
[Cell]
1 0 -1 trcl=(0 0 20) imp:n=1
[Surface]
1 SO 2
"""
    model = load_phits_text(text)

    assert "imp:n=1" in model.cells[0].options
    assert model.geometry.find_cell(0.0, 0.0, 20.0) == 0


def test_bare_inline_trcl_is_rejected():
    text = """title
[Cell]
1 0 -1 trcl=0 0 20
[Surface]
1 SO 2
"""
    with pytest.raises(CellSyntaxError, match="parentheses"):
        load_phits_text(text)


@pytest.mark.parametrize("trcl", ["trcl=()", "trcl=(0 0)", "trcl=(0 0 0 0)"])
def test_invalid_inline_trcl(trcl):
    text = f"""title
[Cell]
1 0 -1 {trcl}
[Surface]
1 SO 2
"""
    with pytest.raises(CellSyntaxError):
        load_phits_text(text)


def test_cell_reference_uses_target_transform():
    text = """title
[Transform]
tr1 10 0 0 1 0 0 0 1 0 0 0 1 1
[Cell]
1 0 -1
2 0 #1
[Surface]
1 1 SO 2
"""
    model = load_phits_text(text)

    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == 1


def test_cell_reference_uses_inline_transform():
    text = """title
[Cell]
1 0 -1 trcl=(10 0 0)
2 0 #1
[Surface]
1 SO 2
"""
    model = load_phits_text(text)

    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == 1


def test_unknown_surface_transform():
    text = """title
[Cell]
1 0 -1
[Surface]
1 2 SO 2
"""
    with pytest.raises(SurfaceSyntaxError, match="unknown transform"):
        load_phits_text(text)


def test_unknown_cell_transform():
    text = """title
[Cell]
1 0 -1 trcl=9
[Surface]
1 SO 2
"""
    with pytest.raises(InputSyntaxError, match="unknown transform"):
        load_phits_text(text)
