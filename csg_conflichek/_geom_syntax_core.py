# Copyright (c) 2026 西安交通大学
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared MCNP/PHITS geometry syntax helpers.

This module is private.  Dialect-specific modules provide the public
entry points and input-section handling.
"""

import re
from dataclasses import dataclass, field
from collections.abc import Mapping

from . import macro
from . import shapes as sp
from . import transforms
from .shapes import Shape
from .cell import (
    Cell,
    CellExpression,
    CellReference,
    Complement,
    Intersection,
    SurfaceTerm,
    Union,
)
from .geometry import CellEntry, Geometry


TOKEN_RE = re.compile(r"[():#]|[^\s():#]+")
SIGNED_INT_RE = re.compile(r"[+-]?\d+\Z")
UNSIGNED_INT_RE = re.compile(r"\d+\Z")


class CellSyntaxError(ValueError):
    """Raised when a cell card or geometry expression cannot be parsed."""


class SurfaceSyntaxError(ValueError):
    """Raised when a surface card cannot be parsed."""


class InputSyntaxError(ValueError):
    """Raised when an input file or section cannot be parsed."""


@dataclass(frozen=True)
class ParsedCell:
    cell_id: int
    material: int
    density: float | None
    expression: CellExpression
    options: tuple[str, ...] = ()
    transform_id: int | None = None
    inline_transform: transforms.Transform | None = None

    def to_cell(self, transform=None):
        return Cell(self.expression, transform=transform)

    @property
    def option_dict(self):
        result = {}
        for token in self.options:
            if "=" in token:
                key, value = token.split("=", 1)
                result[key] = value
        return result


@dataclass(frozen=True)
class ParsedSurface:
    surface_id: int
    mnemonic: str
    shape: object
    transform: int | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedModel:
    syntax: str
    title: str
    surfaces: dict
    cells: tuple
    geometry: Geometry
    transforms: dict = field(default_factory=dict)
    set_variables: dict = field(default_factory=dict)
    set_statements: tuple = ()


class TransformedShape(Shape):
    def __init__(self, shape, transform):
        self.shape = shape
        self.transform = transform

    def is_positive(self, px0, py0, pz0):
        local_x, local_y, local_z = self.transform.global_to_local_xyz(
            px0,
            py0,
            pz0,
        )
        return self.shape.is_positive(local_x, local_y, local_z)

    def __repr__(self):
        return f"TransformedShape({self.shape!r}, {self.transform!r})"


class _ExpressionParser:
    def __init__(self, tokens, surfaces: Mapping[int, object], known_cell_ids=None):
        self.tokens = tokens
        self.surfaces = surfaces
        self.known_cell_ids = known_cell_ids
        self.index = 0

    def parse(self):
        if not self.tokens:
            raise CellSyntaxError("cell geometry is empty")

        expression = self._parse_union()
        if self.index != len(self.tokens):
            raise CellSyntaxError(f"unexpected token {self.tokens[self.index]!r}")
        return expression

    def _peek(self):
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _take(self):
        token = self._peek()
        if token is None:
            raise CellSyntaxError("unexpected end of cell geometry")
        self.index += 1
        return token

    def _parse_union(self):
        branches = [self._parse_intersection()]
        while self._peek() == ":":
            self.index += 1
            branches.append(self._parse_intersection())

        if len(branches) == 1:
            return branches[0]
        return Union(branches)

    def _parse_intersection(self):
        terms = []
        while self._peek() not in (None, ":", ")"):
            terms.append(self._parse_term())

        if not terms:
            raise CellSyntaxError("expected a surface or parenthesized expression")
        if len(terms) == 1:
            return terms[0]
        return Intersection(terms)

    def _parse_term(self):
        token = self._peek()

        if token == "(":
            self.index += 1
            expression = self._parse_union()
            if self._take() != ")":
                raise CellSyntaxError("missing closing parenthesis")
            return expression

        if token == "#":
            self.index += 1
            if self._peek() == "(":
                self.index += 1
                expression = self._parse_union()
                if self._take() != ")":
                    raise CellSyntaxError("missing closing parenthesis")
                return Complement(expression)

            cell_id = _parse_cell_id(self._take())
            if self.known_cell_ids is not None and cell_id not in self.known_cell_ids:
                raise CellSyntaxError(f"unknown cell id: {cell_id}")
            return Complement(CellReference(cell_id))

        return _parse_surface_term(self._take(), self.surfaces)


def _tokenize(text):
    text = text.split("$", 1)[0]
    return TOKEN_RE.findall(text)


def _is_comment_line(line):
    if not line.split("$", 1)[0].strip():
        return True
    return line[0] in ("c", "C") and (len(line) == 1 or line[1].isspace())


def _is_geometry_token(token):
    return token in ("(", ")", ":", "#") or SIGNED_INT_RE.fullmatch(token) is not None


def _parse_surface_term(token, surfaces: Mapping[int, object]):
    if SIGNED_INT_RE.fullmatch(token) is None:
        raise CellSyntaxError(f"invalid surface token: {token!r}")

    value = int(token)
    surface_id = abs(value)
    if surface_id == 0:
        raise CellSyntaxError("surface id must be nonzero")

    try:
        surface = surfaces[surface_id]
    except KeyError:
        raise CellSyntaxError(f"unknown surface id: {surface_id}") from None

    sign = 1 if value > 0 else -1
    return SurfaceTerm(sign, surface)


def _parse_cell_id(token):
    if UNSIGNED_INT_RE.fullmatch(token) is None:
        raise CellSyntaxError(f"invalid cell id: {token!r}")

    cell_id = int(token)
    if cell_id <= 0:
        raise CellSyntaxError("cell id must be positive")
    return cell_id


def _parse_positive_int(token, name):
    if UNSIGNED_INT_RE.fullmatch(token) is None:
        raise CellSyntaxError(f"invalid {name}: {token!r}")

    value = int(token)
    if value <= 0:
        raise CellSyntaxError(f"{name} must be positive")
    return value


def _parse_surface_id(token):
    if UNSIGNED_INT_RE.fullmatch(token) is None:
        raise SurfaceSyntaxError(f"invalid surface id: {token!r}")

    value = int(token)
    if value <= 0:
        raise SurfaceSyntaxError("surface id must be positive")
    return value


def _parse_nonnegative_int(token, name):
    if UNSIGNED_INT_RE.fullmatch(token) is None:
        raise CellSyntaxError(f"invalid {name}: {token!r}")
    return int(token)


def _parse_material(token, allow_negative=False):
    if SIGNED_INT_RE.fullmatch(token) is None:
        raise CellSyntaxError(f"invalid material: {token!r}")
    value = int(token)
    if value < 0 and not allow_negative:
        raise CellSyntaxError("material must be non-negative")
    return value


def _parse_density(token):
    try:
        return float(token)
    except ValueError:
        raise CellSyntaxError(f"invalid material density: {token!r}") from None


_TRCL_START_RE = re.compile(r"(?i)(?<!\S)(\*?)trcl\s*=")
_TRCL_VALUE_RE = re.compile(r"(?i)(?<!\S)(\*?)trcl\s*=\s*(\([^)]*\)|\+?\d+)")
_UNSUPPORTED_CELL_OPTION_RE = re.compile(r"(?i)(?<!\S)(u|lat|fill)\s*=")


def _reject_unsupported_geometry_options(options_text):
    match = _UNSUPPORTED_CELL_OPTION_RE.search(options_text)
    if match:
        raise CellSyntaxError(f"cell parameter {match.group(1)} is not supported yet")


def _parse_transform_option(options_text, allow_inline):
    start = _TRCL_START_RE.search(options_text)
    if start is None:
        return None, None

    match = _TRCL_VALUE_RE.search(options_text)
    if match is None:
        raise CellSyntaxError("unsupported trcl syntax")

    degrees = bool(match.group(1))
    value = match.group(2)

    if not value.startswith("("):
        remainder = options_text[match.end() :]
        if re.match(r"\s*[+-]?(?:\d|\.)", remainder):
            raise CellSyntaxError("inline trcl requires parentheses")
        transform_id = int(value)
        return (transform_id if transform_id else None), None

    if not allow_inline:
        raise CellSyntaxError("inline trcl is not supported in this dialect")

    inner = value[1:-1].strip()
    if not inner:
        raise CellSyntaxError("empty inline trcl")

    try:
        values = [float(item) for item in re.split(r"[\s,]+", inner) if item]
    except ValueError:
        raise CellSyntaxError("invalid inline trcl value") from None

    if len(values) == 3:
        values = values + [1, 0, 0, 0, 1, 0, 0, 0, 1]
        mode = 1
    elif len(values) == 12:
        mode = 1
    elif len(values) == 13:
        mode = int(values[-1])
        values = values[:-1]
    else:
        raise CellSyntaxError("inline trcl needs 3, 12, or 13 values")

    try:
        transform = transforms.Transform.from_phits(
            number=None,
            values=values,
            mode=mode,
            degrees=degrees,
        )
    except ValueError as exc:
        raise CellSyntaxError(f"invalid inline trcl: {exc}") from exc

    return None, transform


_MAT_RE = re.compile(r"(?i)(?<!\S)mat\s*=\s*(\d+)")
_RHO_RE = re.compile(
    r"(?i)(?<!\S)rho\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)"
)


def _is_like_line(tokens):
    return len(tokens) >= 4 and tokens[1].upper() == "LIKE" and tokens[3].upper() == "BUT"


def _parse_like_cell_line(
    line,
    cells_by_id,
    allow_cell_keyword=False,
    allow_inline_transform=False,
):
    clean_line = line.split("$", 1)[0]
    if allow_cell_keyword:
        clean_line = re.sub(
            r"^\s*cell\b",
            "",
            clean_line,
            count=1,
            flags=re.IGNORECASE,
        )

    matches = list(TOKEN_RE.finditer(clean_line))
    tokens = [match.group() for match in matches]
    if not _is_like_line(tokens):
        raise CellSyntaxError("not a LIKE cell line")

    cell_id = _parse_positive_int(tokens[0], "cell id")
    target_id = _parse_positive_int(tokens[2], "LIKE target cell id")
    try:
        base = cells_by_id[target_id]
    except KeyError:
        raise CellSyntaxError(f"LIKE target cell is not defined: {target_id}") from None

    if len(matches) > 4:
        options_text = clean_line[matches[4].start() :]
    else:
        options_text = ""

    _reject_unsupported_geometry_options(options_text)

    material = base.material
    density = base.density
    transform_id = base.transform_id
    inline_transform = base.inline_transform

    mat_match = _MAT_RE.search(options_text)
    if mat_match is not None:
        material = int(mat_match.group(1))

    rho_match = _RHO_RE.search(options_text)
    if rho_match is not None:
        try:
            density = float(rho_match.group(1).replace("D", "e").replace("d", "e"))
        except ValueError:
            raise CellSyntaxError("invalid RHO value in LIKE cell") from None

    if _TRCL_START_RE.search(options_text):
        transform_id, inline_transform = _parse_transform_option(
            options_text,
            allow_inline=allow_inline_transform,
        )

    if material == 0:
        density = None

    options = tuple(options_text.split()) if options_text else ()

    return ParsedCell(
        cell_id=cell_id,
        material=material,
        density=density,
        expression=base.expression,
        options=options,
        transform_id=transform_id,
        inline_transform=inline_transform,
    )


def parse_cell_lines(
    lines,
    surfaces,
    known_cell_ids=None,
    allow_cell_keyword=False,
    allow_inline_transform=False,
    allow_negative_material=False,
):
    cells = []
    cells_by_id = {}

    for line in lines:
        if _is_comment_line(line):
            continue

        token_line = line.split("$", 1)[0]
        if allow_cell_keyword:
            token_line = re.sub(
                r"^\s*cell\b",
                "",
                token_line,
                count=1,
                flags=re.IGNORECASE,
            )
        tokens = TOKEN_RE.findall(token_line)
        is_like = _is_like_line(tokens)

        if is_like:
            parsed = _parse_like_cell_line(
                line,
                cells_by_id,
                allow_cell_keyword=allow_cell_keyword,
                allow_inline_transform=allow_inline_transform,
            )
        else:
            parsed = parse_cell_line(
                line,
                surfaces,
                known_cell_ids=known_cell_ids,
                allow_cell_keyword=allow_cell_keyword,
                allow_inline_transform=allow_inline_transform,
                allow_negative_material=allow_negative_material,
            )

        if parsed.cell_id in cells_by_id:
            raise CellSyntaxError(f"duplicate cell id: {parsed.cell_id}")
        cells_by_id[parsed.cell_id] = parsed
        cells.append(parsed)

    return cells


def parse_expression(text, surfaces: Mapping[int, object], known_cell_ids=None):
    return _ExpressionParser(
        _tokenize(text),
        surfaces,
        known_cell_ids=known_cell_ids,
    ).parse()


def parse_cell_line(
    line,
    surfaces: Mapping[int, object],
    known_cell_ids=None,
    allow_cell_keyword=False,
    allow_inline_transform=False,
    allow_negative_material=False,
):
    clean_line = line.split("$", 1)[0]
    if allow_cell_keyword:
        clean_line = re.sub(
            r"^\s*cell\b",
            "",
            clean_line,
            count=1,
            flags=re.IGNORECASE,
        )

    matches = list(TOKEN_RE.finditer(clean_line))
    tokens = [match.group() for match in matches]

    if len(tokens) < 3:
        raise CellSyntaxError("cell card needs a cell id, material, and geometry")

    cell_id = _parse_positive_int(tokens[0], "cell id")
    material = _parse_material(
        tokens[1],
        allow_negative=allow_negative_material,
    )

    index = 2
    if material == 0:
        density = None
    elif material < 0:
        # PHITS uses negative material numbers for special void-like
        # regions.  The density is optional and is present only when the
        # next token is not a geometry token.
        if index < len(tokens) and not _is_geometry_token(tokens[index]):
            density = _parse_density(tokens[index])
            index += 1
        else:
            density = None
    else:
        if index >= len(tokens):
            raise CellSyntaxError("missing material density")
        density = _parse_density(tokens[index])
        index += 1

    geometry_tokens = []
    while index < len(tokens) and _is_geometry_token(tokens[index]):
        geometry_tokens.append(tokens[index])
        index += 1

    if not geometry_tokens:
        raise CellSyntaxError("cell geometry is empty")

    expression = _ExpressionParser(
        geometry_tokens,
        surfaces,
        known_cell_ids=known_cell_ids,
    ).parse()

    if index < len(matches):
        options_text = clean_line[matches[index].start() :]
        _reject_unsupported_geometry_options(options_text)
        options = tuple(options_text.split())
        transform_id, inline_transform = _parse_transform_option(
            options_text,
            allow_inline=allow_inline_transform,
        )
    else:
        options = ()
        transform_id = None
        inline_transform = None

    return ParsedCell(
        cell_id=cell_id,
        material=material,
        density=density,
        expression=expression,
        options=options,
        transform_id=transform_id,
        inline_transform=inline_transform,
    )


def _float_args(args, mnemonic):
    try:
        return [float(arg) for arg in args]
    except ValueError as exc:
        raise SurfaceSyntaxError(f"{mnemonic}: invalid numeric argument") from exc


def _one_arg(args, mnemonic):
    values = _float_args(args, mnemonic)
    if len(values) != 1:
        raise SurfaceSyntaxError(f"{mnemonic} expects 1 argument")
    return values[0]


def _x_cone(args):
    values = _float_args(args, "K/X")
    if len(values) == 4:
        return sp.XCone(*values)
    if len(values) == 5:
        return sp.XCone(*values[:4], values[4])
    raise ValueError("K/X expects 4 or 5 arguments")


def _y_cone(args):
    values = _float_args(args, "K/Y")
    if len(values) == 4:
        return sp.YCone(*values)
    if len(values) == 5:
        return sp.YCone(*values[:4], values[4])
    raise ValueError("K/Y expects 4 or 5 arguments")


def _z_cone(args):
    values = _float_args(args, "K/Z")
    if len(values) == 4:
        return sp.ZCone(*values)
    if len(values) == 5:
        return sp.ZCone(*values[:4], values[4])
    raise ValueError("K/Z expects 4 or 5 arguments")


def _kx(args):
    values = _float_args(args, "KX")
    if len(values) == 2:
        return sp.KX(*values)
    if len(values) == 3:
        return sp.KX(*values)
    raise ValueError("KX expects 2 or 3 arguments")


def _ky(args):
    values = _float_args(args, "KY")
    if len(values) == 2:
        return sp.KY(*values)
    if len(values) == 3:
        return sp.KY(*values)
    raise ValueError("KY expects 2 or 3 arguments")


def _kz(args):
    values = _float_args(args, "KZ")
    if len(values) == 2:
        return sp.KZ(*values)
    if len(values) == 3:
        return sp.KZ(*values)
    raise ValueError("KZ expects 2 or 3 arguments")


SURFACE_BUILDERS = {
    "P": lambda args: sp.Plane(*_float_args(args, "P")),
    "PX": lambda args: sp.XPlane(1, _one_arg(args, "PX")),
    "PY": lambda args: sp.YPlane(1, _one_arg(args, "PY")),
    "PZ": lambda args: sp.ZPlane(1, _one_arg(args, "PZ")),
    "SO": lambda args: sp.OSphere(*_float_args(args, "SO")),
    "S": lambda args: sp.Sphere(*_float_args(args, "S")),
    "SX": lambda args: sp.XSphere(*_float_args(args, "SX")),
    "SY": lambda args: sp.YSphere(*_float_args(args, "SY")),
    "SZ": lambda args: sp.ZSphere(*_float_args(args, "SZ")),
    "C/X": lambda args: sp.XCylinder(*_float_args(args, "C/X")),
    "C/Y": lambda args: sp.YCylinder(*_float_args(args, "C/Y")),
    "C/Z": lambda args: sp.ZCylinder(*_float_args(args, "C/Z")),
    "CX": lambda args: sp.CX(*_float_args(args, "CX")),
    "CY": lambda args: sp.CY(*_float_args(args, "CY")),
    "CZ": lambda args: sp.CZ(*_float_args(args, "CZ")),
    "K/X": _x_cone,
    "K/Y": _y_cone,
    "K/Z": _z_cone,
    "KX": _kx,
    "KY": _ky,
    "KZ": _kz,
    "SQ": lambda args: sp.SQ(*_float_args(args, "SQ")),
    "GQ": lambda args: sp.General(*_float_args(args, "GQ")),
    "TX": lambda args: sp.XTorus(*_float_args(args, "TX")),
    "TY": lambda args: sp.YTorus(*_float_args(args, "TY")),
    "TZ": lambda args: sp.ZTorus(*_float_args(args, "TZ")),
    "RPP": lambda args: macro.RPP(*_float_args(args, "RPP")),
    "RCC": lambda args: macro.RCC(*_float_args(args, "RCC")),
    "BOX": lambda args: macro.BOX(*_float_args(args, "BOX")),
    "HEX": lambda args: macro.HEX(*_float_args(args, "HEX")),
    "RHP": lambda args: macro.HEX(*_float_args(args, "RHP")),
    "REC": lambda args: macro.REC(*_float_args(args, "REC")),
    "TRC": lambda args: macro.TRC(*_float_args(args, "TRC")),
    "ELL": lambda args: macro.ELL(*_float_args(args, "ELL")),
    "WED": lambda args: macro.WED(*_float_args(args, "WED")),
    "ARB": lambda args: macro.ARB(*args),
    "SPH": lambda args: sp.Sphere(*_float_args(args, "SPH")),
}


def _build_surface(mnemonic, args):
    builder = SURFACE_BUILDERS.get(mnemonic)
    if builder is None:
        raise SurfaceSyntaxError(f"unsupported surface mnemonic: {mnemonic}")

    try:
        return builder(args)
    except SurfaceSyntaxError:
        raise
    except (TypeError, ValueError, IndexError) as exc:
        raise SurfaceSyntaxError(f"{mnemonic}: {exc}") from exc


def parse_surface_line(line):
    clean_line = line.split("$", 1)[0]
    tokens = clean_line.split()

    if len(tokens) < 2:
        raise SurfaceSyntaxError("surface card needs a surface id and a mnemonic")

    surface_id = _parse_surface_id(tokens[0])
    mnemonic_index = 1
    transform = None

    if tokens[1].upper() not in SURFACE_BUILDERS:
        if UNSIGNED_INT_RE.fullmatch(tokens[1]) is None:
            raise SurfaceSyntaxError(f"unknown surface mnemonic: {tokens[1]!r}")
        transform = int(tokens[1])
        mnemonic_index = 2
        if len(tokens) <= mnemonic_index:
            raise SurfaceSyntaxError("missing surface mnemonic")

    mnemonic = tokens[mnemonic_index].upper()
    args = tokens[mnemonic_index + 1 :]
    shape = _build_surface(mnemonic, args)

    return ParsedSurface(
        surface_id=surface_id,
        mnemonic=mnemonic,
        shape=shape,
        transform=transform,
    )


def parse_surface_block_parsed(lines):
    parsed_surfaces = []
    seen_ids = set()

    for line in lines:
        if _is_comment_line(line):
            continue
        parsed = parse_surface_line(line)
        if parsed.surface_id in seen_ids:
            raise SurfaceSyntaxError(f"duplicate surface id: {parsed.surface_id}")
        seen_ids.add(parsed.surface_id)
        parsed_surfaces.append(parsed)

    return parsed_surfaces


def apply_surface_transform(parsed, transforms):
    if parsed.transform in (None, 0):
        return parsed.shape

    try:
        transform = transforms[parsed.transform]
    except KeyError:
        raise SurfaceSyntaxError(f"unknown transform id: {parsed.transform}") from None

    return TransformedShape(parsed.shape, transform)


def parse_surface_block(lines, transforms=None):
    parsed_surfaces = parse_surface_block_parsed(lines)

    if transforms is None:
        if any(parsed.transform not in (None, 0) for parsed in parsed_surfaces):
            raise SurfaceSyntaxError("surface transform requires a transform mapping")
        return {parsed.surface_id: parsed.shape for parsed in parsed_surfaces}

    return {
        parsed.surface_id: apply_surface_transform(parsed, transforms) for parsed in parsed_surfaces
    }


def build_loaded_model(
    syntax,
    title,
    surfaces,
    cells,
    transforms=None,
    set_variables=None,
    set_statements=(),
):
    transforms = transforms or {}
    set_variables = set_variables or {}
    entries = []

    for parsed in cells:
        transform = parsed.inline_transform
        if transform is None and parsed.transform_id:
            try:
                transform = transforms[parsed.transform_id]
            except KeyError:
                raise InputSyntaxError(f"unknown transform id: {parsed.transform_id}") from None

        entries.append(
            CellEntry(
                cell=parsed.to_cell(transform),
                cell_id=parsed.cell_id,
                material=parsed.material,
                density=parsed.density,
            )
        )

    try:
        geometry = Geometry(entries)
    except (TypeError, ValueError) as exc:
        raise InputSyntaxError(str(exc)) from exc

    return LoadedModel(
        syntax=syntax,
        title=title,
        surfaces=surfaces,
        cells=tuple(cells),
        geometry=geometry,
        transforms=transforms,
        set_variables=set_variables,
        set_statements=tuple(set_statements),
    )


class GeomSyntax:
    name = ""
    allow_cell_keyword = False
    allow_inline_transform = False
    allow_negative_material = False

    def parse_surface(self, line):
        return parse_surface_line(line)

    def parse_surfaces(self, lines):
        return parse_surface_block(lines)

    def parse_cell(self, line, surfaces, known_cell_ids=None):
        return parse_cell_line(
            line,
            surfaces,
            known_cell_ids=known_cell_ids,
            allow_cell_keyword=self.allow_cell_keyword,
            allow_inline_transform=self.allow_inline_transform,
            allow_negative_material=self.allow_negative_material,
        )

    def parse_cells(self, lines, surfaces, known_cell_ids=None):
        return parse_cell_lines(
            lines,
            surfaces,
            known_cell_ids=known_cell_ids,
            allow_cell_keyword=self.allow_cell_keyword,
            allow_inline_transform=self.allow_inline_transform,
            allow_negative_material=self.allow_negative_material,
        )

    def load(self, text):
        raise NotImplementedError
