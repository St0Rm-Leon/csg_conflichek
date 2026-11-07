# Copyright (c) 2026 西安交通大学
# SPDX-License-Identifier: GPL-3.0-or-later

"""PHITS ``set:`` expression recognition and Python translation."""

import ast
import math
import re
from dataclasses import dataclass
from collections.abc import Mapping


class SetSyntaxError(ValueError):
    """Raised when a PHITS set expression cannot be parsed or evaluated."""


_SET_LINE_RE = re.compile(r"^\s*set\s*:\s*(.*)$", re.IGNORECASE)
_ASSIGN_RE = re.compile(
    r"(c[1-9]\d{0,2})\s*\[\s*([^\]]*?)\s*\]",
    re.IGNORECASE,
)
_VARIABLE_RE = re.compile(r"c[1-9]\d{0,2}\Z", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_FORTRAN_EXP_RE = re.compile(r"(?<=\d)[dD]([+-]?\d)")
_CHAR_ASSIGN_RE = re.compile(r"%[^%]+%\s*\[")

_MATH_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
}

_MATH_CONSTANTS = {
    "pi": math.pi,
}

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
}

_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
)

_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


@dataclass(frozen=True)
class SetAssignment:
    name: str
    expression: str
    python_expression: str
    value: float
    line_number: int | None = None

    @property
    def python_statement(self):
        return f"{self.name} = {self.python_expression}"


@dataclass(frozen=True)
class SetTranslation:
    text: str
    variables: dict
    assignments: tuple

    @property
    def statements(self):
        return tuple(assignment.python_statement for assignment in self.assignments)


def translate_expression(expression):
    """Translate a PHITS/Fortran-like expression to Python syntax."""
    text = expression.strip()
    text = _FORTRAN_EXP_RE.sub(r"e\1", text)
    text = text.replace("^", "**")

    def replace_identifier(match):
        token = match.group(0)
        lower = token.lower()

        if lower in _MATH_CONSTANTS:
            return f"math.{lower}"
        if lower in _MATH_FUNCTIONS:
            return f"math.{lower}"
        if lower in _SAFE_BUILTINS:
            return lower
        if _VARIABLE_RE.fullmatch(lower):
            return lower

        return token

    return _IDENTIFIER_RE.sub(replace_identifier, text)


def _validate_node(node, variables):
    if isinstance(node, ast.Expression):
        _validate_node(node.body, variables)
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value,
            (int, float),
        ):
            raise SetSyntaxError("invalid constant in set expression")
        return

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        _validate_node(node.left, variables)
        _validate_node(node.right, variables)
        return

    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        _ALLOWED_UNARYOPS,
    ):
        _validate_node(node.operand, variables)
        return

    if isinstance(node, ast.Name):
        if node.id in variables or node.id in _SAFE_BUILTINS:
            return
        raise SetSyntaxError(f"unknown variable or function: {node.id}")

    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and (node.attr in _MATH_FUNCTIONS or node.attr in _MATH_CONSTANTS)
        ):
            return
        raise SetSyntaxError("invalid math attribute in set expression")

    if isinstance(node, ast.Call):
        if node.keywords:
            raise SetSyntaxError("keyword arguments are not supported")

        if isinstance(node.func, ast.Attribute):
            _validate_node(node.func, variables)
        elif isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTINS:
                raise SetSyntaxError(f"unknown function in set expression: {node.func.id}")
        else:
            raise SetSyntaxError("invalid function in set expression")

        for argument in node.args:
            _validate_node(argument, variables)
        return

    raise SetSyntaxError("unsupported syntax in set expression")


def evaluate_expression(python_expression, variables: Mapping[str, float]):
    try:
        tree = ast.parse(python_expression, mode="eval")
    except SyntaxError as exc:
        raise SetSyntaxError(f"invalid set expression: {exc.msg}") from exc

    _validate_node(tree, variables)

    environment = {
        "math": math,
        **_SAFE_BUILTINS,
        **variables,
    }

    try:
        value = eval(
            compile(tree, "<phits-set>", "eval"),
            {"__builtins__": {}},
            environment,
        )
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise SetSyntaxError(f"cannot evaluate set expression: {exc}") from exc

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SetSyntaxError("set expression must evaluate to a number")

    return float(value)


def parse_set_line(line, variables=None, line_number=None):
    clean_line = line.split("$", 1)[0]
    clean_line = re.split(r"[#!]", clean_line, maxsplit=1)[0]
    match = _SET_LINE_RE.match(clean_line)
    if match is None:
        raise SetSyntaxError("not a PHITS set: line")

    payload = match.group(1)
    if _CHAR_ASSIGN_RE.search(payload):
        raise SetSyntaxError("character set variables are not supported")

    if variables is None:
        variables = {}

    assignments = []
    cursor = 0

    for match in _ASSIGN_RE.finditer(payload):
        if payload[cursor : match.start()].strip():
            raise SetSyntaxError("invalid text between set assignments")

        name = match.group(1).lower()
        expression = match.group(2).strip()
        if not expression:
            raise SetSyntaxError(f"empty set expression for {name}")

        python_expression = translate_expression(expression)
        value = evaluate_expression(python_expression, variables)

        assignment = SetAssignment(
            name=name,
            expression=expression,
            python_expression=python_expression,
            value=value,
            line_number=line_number,
        )
        assignments.append(assignment)
        variables[name] = value
        cursor = match.end()

    if payload[cursor:].strip():
        raise SetSyntaxError("invalid trailing text in set: line")
    if not assignments:
        raise SetSyntaxError("set: line contains no assignments")

    return assignments


def _format_number(value):
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _substitute_variables(line, variables):
    code, separator, comment = line.partition("$")

    def replace(match):
        name = f"c{match.group(1)}".lower()
        if name in variables:
            return _format_number(variables[name])
        return match.group(0)

    code = re.sub(r"\bc(\d+)\b", replace, code, flags=re.IGNORECASE)
    return code + separator + comment


def translate_phits_sets(text):
    """Evaluate set: lines and substitute cN variables in later lines."""
    lines = text.splitlines()
    variables = {}
    assignments = []
    output_lines = []

    for line_number, line in enumerate(lines, start=1):
        code = line.split("$", 1)[0]
        if _SET_LINE_RE.match(code):
            assignments.extend(
                parse_set_line(
                    line,
                    variables=variables,
                    line_number=line_number,
                )
            )
            output_lines.append("")
        else:
            output_lines.append(_substitute_variables(line, variables))

    translated_text = "\n".join(output_lines)
    if text.endswith("\n"):
        translated_text += "\n"

    return SetTranslation(
        text=translated_text,
        variables=variables,
        assignments=tuple(assignments),
    )


def translate_set_line(line, variables=None):
    """Translate one set: line to Python statements and evaluated values."""
    assignments = parse_set_line(line, variables=variables)
    return tuple(assignment.python_statement for assignment in assignments)
