# Copyright (c) 2026 西安交通大学
# SPDX-License-Identifier: GPL-3.0-or-later

"""PHITS surface/cell/input syntax interpreter."""

import re

from . import transforms
from .set_syntax import translate_phits_sets
from ._geom_syntax_core import (
    GeomSyntax,
    InputSyntaxError,
    TOKEN_RE,
    UNSIGNED_INT_RE,
    _is_comment_line,
    build_loaded_model,
    parse_cell_line,
    parse_cell_lines,
    parse_expression,
    parse_surface_block,
    parse_surface_line,
)


_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_TRANSFORM_RE = re.compile(r"^(\*?)tr(\d+)$", re.IGNORECASE)


class PHITSGeomSyntax(GeomSyntax):
    name = "phits"
    allow_cell_keyword = True
    allow_inline_transform = True
    allow_negative_material = True

    def load(self, text):
        return load_phits_text(text)


PHITS = PHITSGeomSyntax()


def parse_phits_expression(text, surfaces, known_cell_ids=None):
    return parse_expression(text, surfaces, known_cell_ids)


def parse_phits_surface(line):
    return parse_surface_line(line)


def parse_phits_surfaces(lines, transforms=None):
    return parse_surface_block(lines, transforms)


def parse_phits_cell(line, surfaces, known_cell_ids=None):
    return parse_cell_line(
        line,
        surfaces,
        known_cell_ids=known_cell_ids,
        allow_cell_keyword=True,
        allow_inline_transform=True,
        allow_negative_material=True,
    )


def parse_phits_cells(lines, surfaces, known_cell_ids=None):
    return parse_cell_lines(
        lines,
        surfaces,
        known_cell_ids=known_cell_ids,
        allow_cell_keyword=True,
        allow_inline_transform=True,
        allow_negative_material=True,
    )


def parse_phits_transform_line(line):
    clean_line = line.split("$", 1)[0].strip()
    tokens = clean_line.split()
    if not tokens:
        raise InputSyntaxError("empty PHITS transform card")

    match = _TRANSFORM_RE.fullmatch(tokens[0])
    if match is None:
        raise InputSyntaxError(f"invalid PHITS transform card: {tokens[0]!r}")
    if len(tokens) != 14:
        raise InputSyntaxError("PHITS transform needs a name, 12 values, and M")

    number = int(match.group(2))
    if number <= 0:
        raise InputSyntaxError("transform number must be positive")

    try:
        values = [float(token) for token in tokens[1:13]]
        mode = int(tokens[13])
    except ValueError:
        raise InputSyntaxError("invalid numeric transform parameter") from None

    if mode not in (-3, -2, -1, 1, 2, 3):
        raise InputSyntaxError(f"invalid transform mode: {mode}")

    try:
        return transforms.Transform.from_phits(
            number=number,
            values=values,
            mode=mode,
            degrees=bool(match.group(1)),
        )
    except ValueError as exc:
        raise InputSyntaxError(str(exc)) from exc


def parse_phits_transforms(lines):
    result = {}
    for line in lines:
        stripped = line.split("$", 1)[0].strip()
        if not stripped or _is_comment_line(line):
            continue
        if stripped.lower().startswith("set"):
            continue

        transform = parse_phits_transform_line(line)
        if transform.number in result:
            raise InputSyntaxError(f"duplicate transform number: {transform.number}")
        result[transform.number] = transform
    return result


def _split_phits_sections(text):
    sections = {}
    title_lines = []
    current = None

    for line in text.splitlines():
        stripped = line.split("$", 1)[0].strip()
        match = _SECTION_RE.match(stripped)
        if match:
            current = re.sub(r"\s+", "", match.group(1)).lower()
            sections.setdefault(current, [])
            continue

        if current is None:
            if stripped and not _is_comment_line(line):
                title_lines.append(stripped)
        else:
            sections[current].append(line)

    if "cell" not in sections or "surface" not in sections:
        raise InputSyntaxError("PHITS input needs [Cell] and [Surface] sections")

    title = " ".join(title_lines)
    if "title" in sections:
        section_title = " ".join(
            line.split("$", 1)[0].strip()
            for line in sections["title"]
            if line.split("$", 1)[0].strip() and not _is_comment_line(line)
        )
        if section_title:
            title = section_title
    return title, sections


def _known_cell_ids(lines):
    ids = set()
    for line in lines:
        cleaned = re.sub(
            r"^\s*cell\b",
            "",
            line.split("$", 1)[0],
            count=1,
            flags=re.IGNORECASE,
        )
        tokens = TOKEN_RE.findall(cleaned)
        if tokens and UNSIGNED_INT_RE.fullmatch(tokens[0]):
            ids.add(int(tokens[0]))
    return ids


def load_phits_text(text):
    translation = translate_phits_sets(text)
    title, sections = _split_phits_sections(translation.text)
    cell_lines = sections["cell"]
    surface_lines = sections["surface"]
    transform_lines = sections.get("transform", [])

    transforms_by_id = parse_phits_transforms(transform_lines)
    surfaces = parse_phits_surfaces(surface_lines, transforms_by_id)
    known_ids = _known_cell_ids(cell_lines)
    cells = parse_phits_cells(
        cell_lines,
        surfaces,
        known_cell_ids=known_ids,
    )

    return build_loaded_model(
        "phits",
        title,
        surfaces,
        cells,
        transforms=transforms_by_id,
        set_variables=translation.variables,
        set_statements=translation.statements,
    )
