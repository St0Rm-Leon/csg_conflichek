import numpy as np
import pytest

from csg_conflichek import shapes as sp
from csg_conflichek._geom_syntax_core import TransformedShape
from csg_conflichek.bounds import Bounds, BoundsError, estimate_bounds, shape_bounds
from csg_conflichek.phits_geom_syntax import load_phits_text
from csg_conflichek.transforms import Transform


def test_sphere_bounds():
    bounds = shape_bounds(sp.Sphere(1, 2, 3, 2))

    assert bounds == (1 - 2, 1 + 2, 2 - 2, 2 + 2, 3 - 2, 3 + 2)


def test_rpp_bounds():
    from csg_conflichek import macro

    bounds = shape_bounds(macro.RPP(-1, 2, -3, 4, -5, 6))

    assert bounds == (-1, 2, -3, 4, -5, 6)


def test_torus_bounds():
    bounds = shape_bounds(sp.ZTorus(0, 0, 0, 2, 1, 1))

    assert bounds == (-3, 3, -3, 3, -1, 1)


def test_transformed_shape_bounds():
    shape = TransformedShape(
        sp.OSphere(1),
        Transform(np.eye(3), np.array([10.0, 0.0, 0.0])),
    )
    bounds = shape_bounds(shape)

    assert bounds == pytest.approx((9.0, 11.0, -1.0, 1.0, -1.0, 1.0))


def test_unbounded_shape_returns_none():
    assert shape_bounds(sp.Plane(0, 0, 1, 0)) is None
    assert shape_bounds(sp.ZCylinder(0, 0, 1)) is None


def test_estimate_bounds_uses_finite_surfaces():
    text = """title
[Cell]
1 0 -1
[Surface]
1 SO 2
2 PZ 1
"""
    model = load_phits_text(text)
    bounds = estimate_bounds(model)

    assert bounds == Bounds(-2, 2, -2, 2, -2, 2)


def test_estimate_bounds_applies_margin():
    text = """title
[Cell]
1 0 -1
[Surface]
1 SO 2
"""
    model = load_phits_text(text)
    bounds = estimate_bounds(model, margin=1.0)

    assert bounds == Bounds(-3, 3, -3, 3, -3, 3)


def test_estimate_bounds_includes_cell_transform():
    text = """title
[Transform]
tr1 10 0 0 1 0 0 0 1 0 0 0 1 1
[Cell]
1 0 -1 trcl=1
[Surface]
1 SO 2
"""
    model = load_phits_text(text)
    bounds = estimate_bounds(model)

    assert bounds.xmin == pytest.approx(-2.0)
    assert bounds.xmax == pytest.approx(12.0)


def test_estimate_bounds_requires_finite_surface():
    text = """title
[Cell]
1 0 -1
[Surface]
1 PZ 1
"""
    model = load_phits_text(text)

    with pytest.raises(BoundsError, match="--bounds"):
        estimate_bounds(model)


def test_bounds_rejects_invalid_ranges():
    with pytest.raises(BoundsError):
        Bounds(1, 1, 0, 1, 0, 1)
