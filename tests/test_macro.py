import math

import numpy as np
import pytest

from csg_conflichek import macro as mb


def test_rpp():
    rpp = mb.RPP(-1, 1, -2, 2, -3, 3)

    assert rpp.is_positive(0.0, 0.0, 0.0) is False
    assert rpp.is_positive(2.0, 0.0, 0.0) is True
    assert rpp.is_positive(0.0, 2.1, 0.0) is True
    assert rpp.is_positive(0.0, 0.0, 3.1) is True
    assert rpp.is_positive(1.0, 0.0, 0.0) is False


def test_rcc():
    rcc = mb.RCC(0, 0, 0, 0, 0, 2, 1)

    assert rcc.is_positive(0.0, 0.0, 1.0) is False
    assert rcc.is_positive(0.0, 2.0, 1.0) is True
    assert rcc.is_positive(0.0, 0.0, -0.1) is True
    assert rcc.is_positive(0.0, 0.0, 2.1) is True
    assert rcc.is_positive(1.0, 0.0, 1.0) is False
    assert rcc.is_positive(0.0, 0.0, 0.0) is False
    assert rcc.is_positive(0.0, 0.0, 2.0) is False


def test_box():
    box = mb.BOX(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)

    assert box.is_positive(0.5, 0.5, 0.5) is False
    assert box.is_positive(1.5, 0.5, 0.5) is True
    assert box.is_positive(0.5, 1.5, 0.5) is True
    assert box.is_positive(0.5, 0.5, 1.5) is True
    assert box.is_positive(1.0, 0.5, 0.5) is False
    assert box.is_positive(-0.1, 0.5, 0.5) is True


def _regular_hex():
    return mb.HEX(
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0.5,
        math.sqrt(3) / 2,
        0,
        -0.5,
        math.sqrt(3) / 2,
        0,
    )


def test_hex():
    hexa = _regular_hex()

    assert hexa.is_positive(0.0, 0.0, 0.5) is False
    assert hexa.is_positive(0.9, 0.0, 0.5) is False
    assert hexa.is_positive(1.1, 0.0, 0.5) is True
    assert hexa.is_positive(0.55, 0.9526, 0.5) is True
    assert hexa.is_positive(0.0, 0.0, 1.1) is True
    assert hexa.is_positive(0.0, 0.0, -0.1) is True
    assert hexa.is_positive(1.0, 0.0, 0.5) is False


def test_rhp_is_hex_alias():
    assert mb.RHP is mb.HEX


def test_rec_scalar_minor_radius():
    rec = mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0.5)

    assert rec.is_positive(0.0, 0.0, 1.0) is False
    assert rec.is_positive(0.0, 0.6, 1.0) is True
    assert rec.is_positive(0.0, 0.5, 1.0) is False
    assert rec.is_positive(0.0, 0.0, 2.1) is True
    assert rec.is_positive(0.0, 0.0, 0.0) is False


def test_rec_vector_minor_axis():
    rec = mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0.5, 0)

    assert rec.is_positive(0.0, 0.0, 1.0) is False
    assert rec.is_positive(0.0, 0.6, 1.0) is True
    assert rec.is_positive(0.0, 0.5, 1.0) is False


def test_trc():
    trc = mb.TRC(0, 0, 0, 0, 0, 10, 5, 2)

    assert trc.is_positive(0.0, 0.0, 5.0) is False
    assert trc.is_positive(0.0, 0.0, 0.0) is False
    assert trc.is_positive(0.0, 0.0, 10.0) is False
    assert trc.is_positive(0.0, 0.0, 11.0) is True
    assert trc.is_positive(5.0, 0.0, 0.0) is False
    assert trc.is_positive(5.0, 0.0, 10.0) is True
    assert trc.is_positive(1.0, 0.0, 10.0) is False


def test_ell_foci_form():
    ell = mb.ELL(3, 3, 0, -3, -3, 0, 9)
    axis = math.sqrt(0.5)

    assert ell.is_positive(0.0, 0.0, 0.0) is False
    assert ell.is_positive(8.9 * axis, 8.9 * axis, 0.0) is False
    assert ell.is_positive(10 * axis, 10 * axis, 0.0) is True
    assert ell.is_positive(0.0, 0.0, 7.0) is False
    assert ell.is_positive(0.0, 0.0, 8.0) is True


def test_ell_center_form():
    ell = mb.ELL(0, 0, 0, 6, 6, 0, -6)
    axis = math.sqrt(0.5)

    assert ell.is_positive(0.0, 0.0, 0.0) is False
    assert ell.is_positive(8 * axis, 8 * axis, 0.0) is False
    assert ell.is_positive(9 * axis, 9 * axis, 0.0) is True
    assert ell.is_positive(0.0, 0.0, 5.0) is False
    assert ell.is_positive(0.0, 0.0, 7.0) is True


def test_wed():
    wed = mb.WED(0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 5)

    assert wed.is_positive(1.0, 1.0, 2.0) is False
    assert wed.is_positive(10.0, 1.0, 2.0) is True
    assert wed.is_positive(6.0, 6.0, 2.0) is True
    assert wed.is_positive(0.0, 0.0, -1.0) is True
    assert wed.is_positive(0.0, 0.0, 6.0) is True


def test_arb():
    arb = mb.ARB(
        -5,
        -10,
        -5,
        -5,
        -10,
        5,
        5,
        -10,
        -5,
        5,
        -10,
        5,
        0,
        12,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1234,
        1250,
        1350,
        2450,
        3450,
        0,
    )

    assert arb.is_positive(0.0, 0.0, 0.0) is False
    assert arb.is_positive(0.0, -10.0, 0.0) is False
    assert arb.is_positive(0.0, 13.0, 0.0) is True
    assert arb.is_positive(10.0, 0.0, 0.0) is True
    assert arb.is_positive(0.0, -11.0, 0.0) is True


@pytest.mark.parametrize(
    "shape",
    [
        mb.RPP(-1, 1, -2, 2, -3, 3),
        mb.RCC(0, 0, 0, 0, 0, 2, 1),
        mb.BOX(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        _regular_hex(),
        mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0.5),
    ],
)
def test_macro_scalar_and_array_results_match(shape):
    points = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 3.0],
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
        lambda: mb.RPP(1, -1, 0, 1, 0, 1),
        lambda: mb.RCC(0, 0, 0, 0, 0, 0, 1),
        lambda: mb.RCC(0, 0, 0, 0, 0, 1, 0),
        lambda: mb.BOX(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
        lambda: mb.BOX(0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1),
        lambda: mb.HEX(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1),
        lambda: mb.HEX(0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        lambda: mb.REC(0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5),
        lambda: mb.REC(0, 0, 0, 0, 0, 2, 0, 0, 0, 0.5),
        lambda: mb.REC(0, 0, 0, 0, 0, 2, 0, 0, 1, 0.5),
        lambda: mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0.5, 0.0),
        lambda: mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0),
        lambda: mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0),
        lambda: mb.REC(0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0.5, 0.5),
    ],
)
def test_invalid_macro_parameters(factory):
    with pytest.raises(ValueError):
        factory()
