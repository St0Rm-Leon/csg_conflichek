import numpy as np
import pytest

from csg_conflichek import cell
from csg_conflichek import geometry
from csg_conflichek import shapes as sp
from csg_conflichek.cell import (
    CellReference,
    Complement,
    Intersection,
    SurfaceTerm,
    Union,
    as_expression,
)


def test_surface_term_sign_and_boundary():
    inside = SurfaceTerm(-1, sp.OSphere(1))
    outside = SurfaceTerm(+1, sp.OSphere(1))

    assert bool(inside._evaluate(0.0, 0.0, 0.0)) is True
    assert bool(inside._evaluate(1.0, 0.0, 0.0)) is True
    assert bool(inside._evaluate(2.0, 0.0, 0.0)) is False
    assert bool(outside._evaluate(1.0, 0.0, 0.0)) is False
    assert bool(outside._evaluate(2.0, 0.0, 0.0)) is True


def test_as_expression_accepts_signed_and_raw_shapes():
    signed = as_expression(-sp.OSphere(1))
    raw = as_expression(sp.OSphere(1))

    assert isinstance(signed, SurfaceTerm)
    assert signed.sign == -1
    assert isinstance(raw, SurfaceTerm)
    assert raw.sign == 1


def test_empty_intersection_is_all_space():
    expression = Intersection([])

    assert bool(expression._evaluate(0.0, 0.0, 0.0)) is True
    assert np.array_equal(
        expression._evaluate(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ),
        np.array([True, True]),
    )
    assert cell.Cell(expression).is_inside_cell(0.0, 0.0, 0.0) is True


def test_empty_union_is_empty_space():
    expression = Union([])

    assert bool(expression._evaluate(0.0, 0.0, 0.0)) is False
    assert np.array_equal(
        expression._evaluate(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ),
        np.array([False, False]),
    )
    assert cell.Cell(expression).is_inside_cell(0.0, 0.0, 0.0) is False


def test_intersection_union_complement_operators():
    a = SurfaceTerm(-1, sp.OSphere(1))
    b = SurfaceTerm(-1, sp.Sphere(2, 0, 0, 1))
    c = SurfaceTerm(1, sp.Plane(1, 0, 0, 0))

    union = a | b
    intersection = union & c
    complement = ~a

    # A or B, and on the positive side of x = 0
    assert bool(intersection._evaluate(0.5, 0.0, 0.0)) is True
    assert bool(intersection._evaluate(-0.5, 0.0, 0.0)) is False
    assert bool(intersection._evaluate(2.5, 0.0, 0.0)) is True
    assert bool(intersection._evaluate(3.5, 0.0, 0.0)) is False

    # Complement of the unit sphere
    assert bool(complement._evaluate(0.0, 0.0, 0.0)) is False
    assert bool(complement._evaluate(2.0, 0.0, 0.0)) is True


def test_de_morgan_equivalence():
    a = SurfaceTerm(-1, sp.OSphere(1))
    b = SurfaceTerm(-1, sp.Sphere(2, 0, 0, 1))

    left = ~(a | b)
    right = (~a) & (~b)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    left_values = left._evaluate(points[:, 0], points[:, 1], points[:, 2])
    right_values = right._evaluate(points[:, 0], points[:, 1], points[:, 2])

    assert np.array_equal(left_values, right_values)


def test_legacy_cell_list_and_expression_are_equivalent():
    legacy = cell.Cell([-sp.OSphere(2), +sp.OSphere(1)])
    expression = cell.Cell(
        Intersection(
            [
                SurfaceTerm(-1, sp.OSphere(2)),
                SurfaceTerm(+1, sp.OSphere(1)),
            ]
        )
    )

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ]
    )

    assert np.array_equal(
        legacy.is_inside_cell(points[:, 0], points[:, 1], points[:, 2]),
        expression.is_inside_cell(points[:, 0], points[:, 1], points[:, 2]),
    )


def test_cell_reference_complement_in_geometry():
    space = geometry.Geometry(
        [
            geometry.CellEntry(
                cell.Cell([-sp.OSphere(1)]),
                cell_id=1,
            ),
            geometry.CellEntry(
                cell.Cell(Complement(CellReference(1))),
                cell_id=2,
            ),
        ]
    )

    assert space.find_cell(0.0, 0.0, 0.0) == 0
    assert space.find_cell(2.0, 0.0, 0.0) == 1
    assert space.find_cell(1.0, 0.0, 0.0) == 0


def test_cell_reference_requires_context():
    expression = cell.Cell(Complement(CellReference(1)))

    with pytest.raises(ValueError, match="EvaluationContext"):
        expression.is_inside_cell(0.0, 0.0, 0.0)


def test_unknown_cell_reference():
    space = geometry.Geometry(
        [
            geometry.CellEntry(
                cell.Cell(Complement(CellReference(99))),
                cell_id=1,
            ),
        ]
    )

    with pytest.raises(ValueError, match="unknown cell id"):
        space.sample(0.0, 0.0, 0.0)


def test_circular_cell_reference():
    space = geometry.Geometry(
        [
            geometry.CellEntry(
                cell.Cell(Complement(CellReference(2))),
                cell_id=1,
            ),
            geometry.CellEntry(
                cell.Cell(Complement(CellReference(1))),
                cell_id=2,
            ),
        ]
    )

    with pytest.raises(ValueError, match="circular"):
        space.sample(0.0, 0.0, 0.0)


def test_expression_scalar_and_array_results_match():
    expression = (SurfaceTerm(-1, sp.OSphere(2)) & SurfaceTerm(1, sp.OSphere(1))) | SurfaceTerm(
        -1, sp.Sphere(2, 0, 0, 0.5)
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    array_result = expression._evaluate(points[:, 0], points[:, 1], points[:, 2])
    scalar_result = np.array(
        [bool(expression._evaluate(*point)) for point in points],
        dtype=bool,
    )

    assert np.array_equal(array_result, scalar_result)


@pytest.mark.parametrize("sign", [0, 2, -2])
def test_surface_term_rejects_invalid_sign(sign):
    with pytest.raises(ValueError, match="sign"):
        SurfaceTerm(sign, sp.OSphere(1))
