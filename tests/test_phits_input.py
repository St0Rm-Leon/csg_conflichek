# Copyright (c) 2026 csg_conflichek contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

import pytest

from csg_conflichek.phits_geom_syntax import load_phits_text


REPO_ROOT = Path(__file__).resolve().parents[1]
PHITS_PATH = REPO_ROOT / "tests" / "data" / "phits.inp"


def test_spaced_section_headers_and_title():
    text = """ignored
[ T i t l e ]
Spaced title
[ C e l l ]
1 0 -1
[ S u r f a c e ]
1 SO 2
"""

    model = load_phits_text(text)

    assert model.title == "Spaced title"
    assert model.geometry.cell_ids == [1]
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == 0


def test_negative_material_without_density():
    text = """title
[Cell]
2 -1 1
[Surface]
1 SO 2
"""

    model = load_phits_text(text)

    assert model.cells[0].material == -1
    assert model.cells[0].density is None
    assert model.geometry.find_cell(3.0, 0.0, 0.0) == 0


def test_negative_material_with_density():
    text = """title
[Cell]
2 -1 1.5 -1
[Surface]
1 SO 2
"""

    model = load_phits_text(text)

    assert model.cells[0].material == -1
    assert model.cells[0].density == 1.5
    assert model.geometry.find_cell(0.0, 0.0, 0.0) == 0


@pytest.mark.skipif(not PHITS_PATH.is_file(), reason="input/phits.inp is missing")
def test_repository_phits_input_loads():
    model = load_phits_text(PHITS_PATH.read_text(encoding="utf-8"))

    assert model.syntax == "phits"
    assert model.title.startswith("Dose in a 40MeV")
    assert len(model.surfaces) == 12
    assert len(model.geometry) == 11
    assert model.geometry.cell_ids == [
        104,
        100,
        110,
        111,
        105,
        106,
        101,
        102,
        103,
        900,
        901,
    ]
    assert model.cells[-1].material == -1
    assert model.cells[-1].density is None


@pytest.mark.skipif(not PHITS_PATH.is_file(), reason="input/phits.inp is missing")
def test_repository_phits_sample_points():
    model = load_phits_text(PHITS_PATH.read_text(encoding="utf-8"))

    checks = [
        ((0.0, 0.0, 0.0), 104),
        ((3.0, 0.0, 0.0), 100),
        ((5.0, 0.0, 0.0), 110),
        ((10.0, 0.0, 0.0), 111),
        ((12.25, 0.0, 0.0), 101),
        ((0.0, 0.0, 5.0), 106),
        ((0.0, 0.0, -5.0), 105),
        ((0.0, 0.0, 12.25), 103),
        ((0.0, 0.0, -12.25), 102),
        ((0.0, 0.0, 14.0), 900),
        ((60.0, 0.0, 0.0), 901),
    ]

    for point, expected_cell_id in checks:
        index = model.geometry.find_cell(*point)
        assert model.geometry.cell_ids[index] == expected_cell_id
