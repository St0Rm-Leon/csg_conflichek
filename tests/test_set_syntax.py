import math

import pytest

from csg_conflichek.phits_geom_syntax import load_phits_text
from csg_conflichek.set_syntax import (
    SetSyntaxError,
    evaluate_expression,
    parse_set_line,
    translate_expression,
    translate_phits_sets,
    translate_set_line,
)


def test_translate_expression_functions_and_constants():
    assert translate_expression("2 * pi") == "2 * math.pi"
    assert translate_expression("cos(c1/180*pi)") == "math.cos(c1/180*math.pi)"
    assert translate_expression("1.0D-3 + 2^2") == "1.0e-3 + 2**2"


def test_parse_set_line_multiple_assignments():
    variables = {}
    assignments = parse_set_line(
        "set: c1[ 52.3 ] c2[ 2 * pi ] c3[ c1 * 1.e-8 ]",
        variables=variables,
        line_number=3,
    )

    assert [assignment.name for assignment in assignments] == [
        "c1",
        "c2",
        "c3",
    ]
    assert variables["c1"] == pytest.approx(52.3)
    assert variables["c2"] == pytest.approx(2 * math.pi)
    assert variables["c3"] == pytest.approx(52.3e-8)
    assert assignments[1].python_statement == "c2 = 2 * math.pi"
    assert assignments[2].line_number == 3


def test_evaluate_expression_with_math():
    value = evaluate_expression(
        "math.sqrt(c1) + math.sin(math.pi / 2)",
        {"c1": 9.0},
    )

    assert value == pytest.approx(4.0)


def test_translate_set_line_returns_python_statements():
    statements = translate_set_line(
        "set: c1[2] c2[c1 + 3]",
    )

    assert statements == ("c1 = 2", "c2 = c1 + 3")


def test_translate_phits_sets_substitutes_variables():
    text = """set: c1[10] c2[2]
[Surface]
1 SO c1
[Cell]
1 0 -1
"""
    translation = translate_phits_sets(text)

    assert "1 SO 10" in translation.text
    assert "c1" not in translation.text
    assert translation.variables == {"c1": 10.0, "c2": 2.0}
    assert translation.statements == ("c1 = 10", "c2 = 2")


def test_load_phits_with_set_variables():
    text = """title
[Parameters]
set: c1[10] c2[2]
[Transform]
tr1 c1 0 0 1 0 0 0 1 0 0 0 1 1
[Cell]
1 0 -1
[Surface]
1 1 SO c2
"""
    model = load_phits_text(text)

    assert model.set_variables == {"c1": 10.0, "c2": 2.0}
    assert model.set_statements == ("c1 = 10", "c2 = 2")
    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == -1


def test_set_variables_in_inline_trcl():
    text = """title
set: c1[10]
[Cell]
1 0 -1 trcl=(c1 0 0)
[Surface]
1 SO 2
"""
    model = load_phits_text(text)

    assert model.geometry.find_cell(10.0, 0.0, 0.0) == 0


def test_set_line_supports_comments():
    translation = translate_phits_sets("set: c1[2] c2[3] # comment\nset: c3[c1+c2] $ comment\n")

    assert translation.variables == {
        "c1": 2.0,
        "c2": 3.0,
        "c3": 5.0,
    }


def test_character_set_variables_are_rejected():
    with pytest.raises(SetSyntaxError, match="character"):
        parse_set_line("set: %part name%[neutron proton]")


@pytest.mark.parametrize(
    "line",
    [
        "set:",
        "set: c1[]",
        "set: c1000[1]",
        "set: c1[1] garbage",
        "set: c1[__import__('os')]",
    ],
)
def test_invalid_set_lines(line):
    with pytest.raises(SetSyntaxError):
        parse_set_line(line)


def test_set_variable_must_be_defined_before_use():
    text = """title
[Cell]
1 0 -1
[Surface]
1 SO c1
set: c1[2]
"""
    with pytest.raises(Exception):
        load_phits_text(text)
