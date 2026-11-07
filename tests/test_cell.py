import numpy as np
import pytest

from csg_conflichek import cell
from csg_conflichek import shapes as sp


def test_scalar_and_array_results_match():
    c = cell.Cell([-sp.OSphere(2), +sp.Plane(1, 0, 0, 0.5)])
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.6, 2.0, 0.0],
            [1.0, 0.5, 0.5],
        ]
    )

    array_result = c.is_inside_cell(points[:, 0], points[:, 1], points[:, 2])
    scalar_result = np.array(
        [c.is_inside_cell(*point) for point in points],
        dtype=bool,
    )

    assert np.array_equal(array_result, scalar_result)


def test_random_scalar_and_array_results_match():
    c = cell.Cell(
        [
            +sp.Plane(1, 0, 0, 0.5),
            -sp.OSphere(2),
            -sp.ZCylinder(0, 0, 1),
        ]
    )
    points = np.random.default_rng(1234).uniform(-3.0, 3.0, size=(200, 3))

    array_result = c.is_inside_cell(points[:, 0], points[:, 1], points[:, 2])
    scalar_result = np.array(
        [c.is_inside_cell(*point) for point in points],
        dtype=bool,
    )

    assert np.array_equal(array_result, scalar_result)


def test_negative_sphere_means_inside():
    c = cell.Cell([-sp.OSphere(1)])

    assert c.is_inside_cell(0.0, 0.0, 0.0) is True
    assert c.is_inside_cell(0.5, 0.0, 0.0) is True
    assert c.is_inside_cell(2.0, 0.0, 0.0) is False


def test_raw_shape_defaults_to_positive_side():
    c = cell.Cell([sp.OSphere(1)])

    assert c.is_inside_cell(0.0, 0.0, 0.0) is False
    assert c.is_inside_cell(2.0, 0.0, 0.0) is True


def test_boundary_belongs_to_negative_side():
    assert cell.Cell([-sp.OSphere(1)]).is_inside_cell(1.0, 0.0, 0.0) is True
    assert cell.Cell([+sp.OSphere(1)]).is_inside_cell(1.0, 0.0, 0.0) is False


def test_plane_boundary_belongs_to_negative_side():
    plane = sp.Plane(1, 0, 0, 0.5)

    assert cell.Cell([-plane]).is_inside_cell(0.5, 0.0, 0.0) is True
    assert cell.Cell([+plane]).is_inside_cell(0.5, 0.0, 0.0) is False


def test_spherical_shell():
    c = cell.Cell([-sp.OSphere(2), +sp.OSphere(1)])

    assert c.is_inside_cell(1.5, 0.0, 0.0) is True
    assert c.is_inside_cell(0.5, 0.0, 0.0) is False
    assert c.is_inside_cell(3.0, 0.0, 0.0) is False
    assert c.is_inside_cell(2.0, 0.0, 0.0) is True
    assert c.is_inside_cell(1.0, 0.0, 0.0) is False


def test_three_surface_intersection():
    c = cell.Cell(
        [
            +sp.Plane(1, 0, 0, 0.5),
            -sp.OSphere(2),
            -sp.ZCylinder(0, 0, 1),
        ]
    )

    assert c.is_inside_cell(1.0, 0.0, 0.0) is True
    assert c.is_inside_cell(1.0, 0.0, 1.0) is True
    assert c.is_inside_cell(0.0, 0.0, 0.0) is False
    assert c.is_inside_cell(2.0, 0.0, 0.0) is False
    assert c.is_inside_cell(1.0, 0.0, 2.0) is False


def test_empty_cell_contains_everything():
    c = cell.Cell([])

    assert c.is_inside_cell(0.0, 0.0, 0.0) is True
    assert np.array_equal(
        c.is_inside_cell(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ),
        np.array([True, True]),
    )


def test_broadcasting_scalar_and_array_inputs():
    c = cell.Cell([-sp.OSphere(1)])

    assert np.array_equal(
        c.is_inside_cell(0.0, np.array([0.0, 2.0]), 0.0),
        np.array([True, False]),
    )
    assert np.array_equal(
        c.is_inside_cell(np.array([[0.0], [2.0]]), 0.0, 0.0),
        np.array([[True], [False]]),
    )
    assert np.array_equal(
        c.is_inside_cell(np.array([[0.0], [2.0]]), np.array([0.0, 0.0]), 0.0),
        np.array([[True, True], [False, False]]),
    )


def test_return_types():
    c = cell.Cell([-sp.OSphere(1)])

    assert isinstance(c.is_inside_cell(0.0, 0.0, 0.0), bool)
    assert isinstance(
        c.is_inside_cell(np.array([0.0]), np.array([0.0]), np.array([0.0])),
        np.ndarray,
    )


@pytest.mark.parametrize("sign", [0, 2, -2])
def test_signed_shape_rejects_invalid_sign(sign):
    with pytest.raises(ValueError, match="sign"):
        sp.SignedShape(sign, sp.OSphere(1))
