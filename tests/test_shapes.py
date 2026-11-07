import numpy as np
import pytest

from csg_conflichek import shapes as sp


def test_plane_uses_strict_positive_side():
    plane = sp.Plane(1, 0, 0, 0.5)

    assert plane.is_positive(1.0, 0.0, 0.0) is True
    assert plane.is_positive(0.0, 0.0, 0.0) is False
    assert plane.is_positive(0.5, 0.0, 0.0) is False


@pytest.mark.parametrize(
    "plane, inside, outside",
    [
        (sp.XPlane(1, 0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (sp.XPlane(-1, 5), (-6.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
        (sp.YPlane(1, 0.5), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        (sp.YPlane(-1, 5), (0.0, -6.0, 0.0), (0.0, -4.0, 0.0)),
        (sp.ZPlane(1, 0.5), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        (sp.ZPlane(-1, 5), (0.0, 0.0, -6.0), (0.0, 0.0, -4.0)),
    ],
)
def test_axis_planes_respect_coefficient_sign(plane, inside, outside):
    assert plane.is_positive(*inside) is True
    assert plane.is_positive(*outside) is False


def test_axis_planes_use_strict_boundary():
    assert sp.XPlane(1, 0.0).is_positive(0.0, 0.0, 0.0) is False
    assert sp.YPlane(1, 0.0).is_positive(0.0, 0.0, 0.0) is False
    assert sp.ZPlane(1, 0.0).is_positive(0.0, 0.0, 0.0) is False


def test_sphere_positive_sense_and_boundary():
    sphere = sp.Sphere(1, 2, 3, 2)

    assert sphere.is_positive(1, 2, 3) is False
    assert sphere.is_positive(1, 2, 4.9) is False
    assert sphere.is_positive(1, 2, 5.0) is False
    assert sphere.is_positive(1, 2, 5.1) is True


@pytest.mark.parametrize(
    "sphere, center, outside",
    [
        (sp.OSphere(2), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        (sp.XSphere(1, 2), (1.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        (sp.YSphere(1, 2), (0.0, 1.0, 0.0), (0.0, 4.0, 0.0)),
        (sp.ZSphere(1, 2), (0.0, 0.0, 1.0), (0.0, 0.0, 4.0)),
    ],
)
def test_axis_spheres(sphere, center, outside):
    assert sphere.is_positive(*center) is False
    assert sphere.is_positive(*outside) is True


@pytest.mark.parametrize(
    "cylinder, inside, outside",
    [
        (sp.XCylinder(0, 0, 1), (0.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        (sp.YCylinder(0, 0, 1), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        (sp.ZCylinder(0, 0, 1), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        (sp.CX(1), (0.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        (sp.CY(1), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        (sp.CZ(1), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    ],
)
def test_cylinders_and_axis_convenience_classes(cylinder, inside, outside):
    assert cylinder.is_positive(*inside) is False
    assert cylinder.is_positive(*outside) is True
    assert cylinder.is_positive(*[0.0, 0.0, 0.0]) is False


def test_cylinder_boundary_is_not_positive():
    assert sp.CZ(1).is_positive(1.0, 0.0, 0.0) is False


@pytest.mark.parametrize(
    "cone, inside, outside, wrong_sheet",
    [
        (sp.XCone(0, 0, 0, 1, 1), (2.0, 0.0, 0.0), (1.0, 2.0, 0.0), (-2.0, 0.0, 0.0)),
        (sp.YCone(0, 0, 0, 1, 1), (0.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, -2.0, 0.0)),
        (sp.ZCone(0, 0, 0, 1, 1), (0.0, 0.0, 2.0), (2.0, 0.0, 1.0), (0.0, 0.0, -2.0)),
        (sp.KX(0, 1, 1), (2.0, 0.0, 0.0), (1.0, 2.0, 0.0), (-2.0, 0.0, 0.0)),
        (sp.KY(0, 1, 1), (0.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, -2.0, 0.0)),
        (sp.KZ(0, 1, 1), (0.0, 0.0, 2.0), (2.0, 0.0, 1.0), (0.0, 0.0, -2.0)),
    ],
)
def test_cone_sheets(cone, inside, outside, wrong_sheet):
    assert cone.is_positive(*inside) is False
    assert cone.is_positive(*outside) is True
    assert cone.is_positive(*wrong_sheet) is True


def test_cone_negative_sheet():
    cone = sp.KZ(0, 1, -1)

    assert cone.is_positive(0.0, 0.0, -2.0) is False
    assert cone.is_positive(0.0, 0.0, 2.0) is True


def test_cone_boundary_is_not_positive():
    cone = sp.KX(0, 1, 1)

    assert cone.is_positive(1.0, 1.0, 0.0) is False


def test_sq_shifted_sphere():
    sq = sp.SQ(1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    assert sq.is_positive(-1, 0, 0) is False
    assert sq.is_positive(0, 0, 0) is False
    assert sq.is_positive(-2, 0, 0) is False
    assert sq.is_positive(-3, 0, 0) is True


def test_general_sphere():
    gq = sp.General(1, 1, 1, 0, 0, 0, 0, 0, 0, -1)

    assert gq.is_positive(0, 0, 0) is False
    assert gq.is_positive(1, 0, 0) is False
    assert gq.is_positive(2, 0, 0) is True


@pytest.mark.parametrize(
    "torus, inside, outside",
    [
        (sp.XTorus(0, 0, 0, 2, 1, 1), (0.0, 2.0, 0.0), (1.1, 2.0, 0.0)),
        (sp.YTorus(0, 0, 0, 2, 1, 1), (2.0, 0.0, 0.0), (2.0, 1.1, 0.0)),
        (sp.ZTorus(0, 0, 0, 2, 1, 1), (2.0, 0.0, 0.0), (2.0, 0.0, 1.1)),
    ],
)
def test_torus(torus, inside, outside):
    assert bool(torus.is_positive(*inside)) is False
    assert bool(torus.is_positive(*outside)) is True


def test_torus_boundary_is_not_positive():
    assert bool(sp.ZTorus(0, 0, 0, 2, 1, 1).is_positive(3.0, 0.0, 0.0)) is False


@pytest.mark.parametrize(
    "shape",
    [
        sp.Plane(1, 0, 0, 0.5),
        sp.OSphere(2),
        sp.ZCylinder(0, 0, 1),
        sp.KX(0, 1, 1),
        sp.SQ(1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
        sp.General(1, 1, 1, 0, 0, 0, 0, 0, 0, -1),
        sp.ZTorus(0, 0, 0, 2, 1, 1),
    ],
)
def test_scalar_and_array_results_match(shape):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
        ]
    )

    array_result = shape.is_positive(points[:, 0], points[:, 1], points[:, 2])
    scalar_result = np.array(
        [shape.is_positive(*point) for point in points],
        dtype=bool,
    )

    assert np.array_equal(array_result, scalar_result)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sp.Plane(0, 0, 0, 1),
        lambda: sp.XPlane(0, 1),
        lambda: sp.YPlane(0, 1),
        lambda: sp.ZPlane(0, 1),
        lambda: sp.OSphere(0),
        lambda: sp.XCylinder(0, 0, 0),
        lambda: sp.YCylinder(0, 0, 0),
        lambda: sp.ZCylinder(0, 0, 0),
        lambda: sp.XCone(0, 0, 0, 0),
        lambda: sp.KX(0, 1, 0),
        lambda: sp.ZTorus(0, 0, 0, 0, 1, 1),
        lambda: sp.SQ(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        lambda: sp.General(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ],
)
def test_invalid_surface_parameters(factory):
    with pytest.raises(ValueError):
        factory()
