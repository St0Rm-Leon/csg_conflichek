import csv
import subprocess
import sys
from pathlib import Path

import pytest

from csg_conflichek.cli import main, parse_size


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


def _rows(path):
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_parse_size():
    assert parse_size("200") == (200, 200, 200)
    assert parse_size("20,30,40") == (20, 30, 40)
    assert parse_size("20x30x40") == (20, 30, 40)

    with pytest.raises(ValueError):
        parse_size("1")
    with pytest.raises(ValueError):
        parse_size("20,30")


def test_cli_reports_no_conflicts(tmp_path):
    output = tmp_path / "simple.csv"

    code = main(
        [
            "-i",
            str(EXAMPLES / "simple_phits.inp"),
            "-o",
            str(output),
            "--size",
            "20",
        ]
    )

    assert code == 0
    assert output.read_text(encoding="utf-8").splitlines() == ["no,type,x,y,z,cell1,cell2"]


def test_cli_writes_overlaps(tmp_path):
    output = tmp_path / "overlap.csv"

    code = main(
        [
            "-i",
            str(EXAMPLES / "overlap_phits.inp"),
            "-o",
            str(output),
            "--size",
            "20",
        ]
    )

    assert code == 0
    rows = _rows(output)
    assert rows
    assert all(row["type"] in {"overlap", "undefined"} for row in rows)
    assert any(row["type"] == "overlap" for row in rows)


def test_cli_writes_undefined_with_explicit_bounds(tmp_path):
    output = tmp_path / "undefined.csv"

    code = main(
        [
            "-i",
            str(EXAMPLES / "undefined_phits.inp"),
            "-o",
            str(output),
            "--size",
            "10",
            "--bounds",
            "-3",
            "3",
            "-3",
            "3",
            "-3",
            "3",
        ]
    )

    assert code == 0
    rows = _rows(output)
    assert rows
    assert any(row["type"] == "undefined" for row in rows)


def test_cli_fail_on_conflict(tmp_path):
    output = tmp_path / "overlap.csv"

    code = main(
        [
            "-i",
            str(EXAMPLES / "overlap_phits.inp"),
            "-o",
            str(output),
            "--size",
            "20",
            "--fail-on-conflict",
        ]
    )

    assert code == 1


def test_cli_no_merge(tmp_path):
    merged_output = tmp_path / "merged.csv"
    raw_output = tmp_path / "raw.csv"

    args = ["-i", str(EXAMPLES / "overlap_phits.inp"), "--size", "20"]
    assert main([*args, "-o", str(merged_output)]) == 0
    assert main([*args, "-o", str(raw_output), "--no-merge"]) == 0

    assert len(_rows(raw_output)) >= len(_rows(merged_output))


def test_cli_workers(tmp_path):
    output = tmp_path / "workers.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "csg_conflichek.cli",
            "-i",
            str(EXAMPLES / "overlap_phits.inp"),
            "-o",
            str(output),
            "--size",
            "20",
            "--workers",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = _rows(output)
    assert any(row["type"] == "overlap" for row in rows)


def test_cli_verify(tmp_path, capsys):
    output = tmp_path / "verify.csv"

    code = main(
        [
            "-i",
            str(EXAMPLES / "overlap_phits.inp"),
            "-o",
            str(output),
            "--size",
            "10",
            "--verify",
            "--verify-size",
            "15",
            "--workers",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "verify: 10x10x10 -> 15x15x15" in captured.out
    assert "overlap volume:" in captured.out
    assert "undefined volume:" in captured.out


def test_cli_missing_input(tmp_path):
    code = main(["-i", str(tmp_path / "missing.inp"), "-o", str(tmp_path / "out.csv")])

    assert code == 1


def test_cli_auto_bounds_failure(tmp_path):
    input_path = tmp_path / "planes.inp"
    input_path.write_text(
        "title\n[Cell]\n1 0 -1\n[Surface]\n1 PZ 1\n",
        encoding="utf-8",
    )

    code = main(["-i", str(input_path), "-o", str(tmp_path / "out.csv"), "--size", "10"])

    assert code == 1
