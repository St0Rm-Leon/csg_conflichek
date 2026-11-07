import csv
from types import SimpleNamespace

import pytest

from csg_conflichek import shapes as sp
from csg_conflichek.bounds import Bounds
from csg_conflichek.cell import Cell
from csg_conflichek.conflict import (
    OVERLAP,
    UNDEFINED,
    ConflictBlock,
    ConflictPoint,
    count_by_type,
    find_conflict_points,
    merge_conflict_points,
    scan_conflicts,
    write_conflicts_csv,
)
from csg_conflichek.geometry import CellEntry, Geometry


def _overlap_model():
    geometry = Geometry(
        [
            CellEntry(Cell([-sp.OSphere(2)]), cell_id=1),
            CellEntry(Cell([-sp.Sphere(1, 0, 0, 2)]), cell_id=2),
        ]
    )
    return SimpleNamespace(geometry=geometry)


def test_find_overlap_points():
    model = _overlap_model()
    bounds = Bounds(-3, 3, -3, 3, -3, 3)
    points = find_conflict_points(model, bounds, (20, 20, 20))

    overlaps = [point for point in points if point.type == OVERLAP]
    assert overlaps
    assert any(point.cell_ids == (1, 2) for point in overlaps)


def test_find_undefined_points():
    model = SimpleNamespace(geometry=Geometry([CellEntry(Cell([-sp.OSphere(1)]), cell_id=1)]))
    bounds = Bounds(-3, 3, -3, 3, -3, 3)
    points = find_conflict_points(model, bounds, (10, 10, 10))

    undefined = [point for point in points if point.type == UNDEFINED]
    assert undefined
    assert all(point.cell_ids == () for point in undefined)
    assert all(point.type == UNDEFINED for point in undefined)


def test_merge_adjacent_points_into_one_block():
    points = [
        ConflictPoint(0, 0, 0, 0.0, 0.0, 0.0, OVERLAP, (1, 2)),
        ConflictPoint(1, 0, 0, 1.0, 0.0, 0.0, OVERLAP, (1, 2)),
        ConflictPoint(0, 1, 0, 0.0, 1.0, 0.0, OVERLAP, (1, 2)),
        ConflictPoint(1, 1, 0, 1.0, 1.0, 0.0, OVERLAP, (1, 2)),
    ]

    blocks = merge_conflict_points(points)

    assert len(blocks) == 1
    assert blocks[0].point_count == 4
    assert blocks[0].x == pytest.approx(0.5)
    assert blocks[0].y == pytest.approx(0.5)
    assert blocks[0].z == pytest.approx(0.0)
    assert blocks[0].cell_ids == (1, 2)


def test_merge_keeps_different_claims_separate():
    points = [
        ConflictPoint(0, 0, 0, 0.0, 0.0, 0.0, OVERLAP, (1, 2)),
        ConflictPoint(1, 0, 0, 1.0, 0.0, 0.0, OVERLAP, (1, 3)),
    ]

    blocks = merge_conflict_points(points)

    assert len(blocks) == 2
    assert {block.cell_ids for block in blocks} == {(1, 2), (1, 3)}


def test_merge_does_not_connect_diagonal_points():
    points = [
        ConflictPoint(0, 0, 0, 0.0, 0.0, 0.0, OVERLAP, (1, 2)),
        ConflictPoint(1, 1, 0, 1.0, 1.0, 0.0, OVERLAP, (1, 2)),
    ]

    blocks = merge_conflict_points(points)

    assert len(blocks) == 2


def test_scan_without_merge_keeps_every_point():
    model = _overlap_model()
    bounds = Bounds(-3, 3, -3, 3, -3, 3)

    points = find_conflict_points(model, bounds, (10, 10, 10))
    blocks = scan_conflicts(model, bounds, (10, 10, 10), merge=False)

    assert len(blocks) == len(points)
    assert all(block.point_count == 1 for block in blocks)


def test_optimized_merge_matches_point_merge():
    model = _overlap_model()
    bounds = Bounds(-3, 3, -3, 3, -3, 3)
    size = (20, 20, 20)

    points = find_conflict_points(model, bounds, size)
    expected = merge_conflict_points(points)
    actual = scan_conflicts(model, bounds, size, workers=1)

    assert len(actual) == len(expected)
    for actual_block, expected_block in zip(actual, expected):
        assert actual_block.type == expected_block.type
        assert actual_block.cell_ids == expected_block.cell_ids
        assert actual_block.point_count == expected_block.point_count
        assert actual_block.x == pytest.approx(expected_block.x)
        assert actual_block.y == pytest.approx(expected_block.y)
        assert actual_block.z == pytest.approx(expected_block.z)


def test_write_csv_schema(tmp_path):
    blocks = [
        ConflictBlock(OVERLAP, 0.0, 0.0, 0.0, (1, 2), 3),
        ConflictBlock(OVERLAP, 1.0, 1.0, 1.0, (1, 2, 3), 4),
        ConflictBlock(UNDEFINED, 2.0, 2.0, 2.0, (), 1),
    ]
    path = tmp_path / "conflicts.csv"

    write_conflicts_csv(blocks, path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))

    assert path.read_text(encoding="utf-8").splitlines()[0] == ("no,type,x,y,z,cell1,cell2")
    assert rows[0]["no"] == "1"
    assert rows[0]["cell1"] == "1"
    assert rows[0]["cell2"] == "2"
    assert rows[1]["cell1"] == "1;2;3"
    assert rows[1]["cell2"] == ""
    assert rows[2]["type"] == UNDEFINED
    assert rows[2]["cell1"] == ""
    assert rows[2]["cell2"] == ""


def test_count_by_type():
    blocks = [
        ConflictBlock(OVERLAP, 0.0, 0.0, 0.0, (1, 2)),
        ConflictBlock(OVERLAP, 1.0, 0.0, 0.0, (2, 3)),
        ConflictBlock(UNDEFINED, 2.0, 0.0, 0.0, ()),
    ]

    assert count_by_type(blocks) == (2, 1)


def test_more_than_64_cells_use_multiple_code_words():
    entries = [CellEntry(Cell([-sp.OSphere(2)]), cell_id=index) for index in range(1, 71)]
    model = SimpleNamespace(geometry=Geometry(entries))

    blocks = scan_conflicts(
        model,
        Bounds(-2, 2, -2, 2, -2, 2),
        (10, 10, 10),
        workers=1,
    )

    overlaps = [block for block in blocks if block.type == OVERLAP]
    assert len(overlaps) == 1
    assert overlaps[0].cell_ids == tuple(range(1, 71))
    assert overlaps[0].point_count == 360
