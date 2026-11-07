# Copyright (c) 2026 西安交通大学
# SPDX-License-Identifier: GPL-3.0-or-later

import csv
import json
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .cell import Cell, EvaluationContext


MULTI_DEFINED = -2


@dataclass(frozen=True)
class CellEntry:
    cell: Cell
    cell_id: int
    material: int | None = None
    color: str | None = None
    density: float | None = None


class Geometry:
    """An ordered collection of CSG cells.

    Cell order is significant: sample() uses first-match semantics, so
    earlier cells take precedence over later cells.
    """

    def __init__(self, entries: Iterable[CellEntry | Cell]):
        resolved: list[CellEntry] = []

        for index, item in enumerate(entries):
            if isinstance(item, CellEntry):
                entry = item
            elif isinstance(item, Cell):
                entry = CellEntry(cell=item, cell_id=index)
            else:
                raise TypeError("Geometry entries must be Cell or CellEntry instances")
            resolved.append(entry)

        cell_ids = [entry.cell_id for entry in resolved]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("cell_id values must be unique")

        self.entries = resolved

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def cell_ids(self):
        return [entry.cell_id for entry in self.entries]

    def claim_masks(self, px0, py0, pz0):
        """Return one boolean membership mask per cell."""
        px0, py0, pz0 = np.broadcast_arrays(px0, py0, pz0)
        masks = np.empty((len(self.entries), *px0.shape), dtype=bool)
        context = EvaluationContext({entry.cell_id: entry.cell for entry in self.entries})

        for index, entry in enumerate(self.entries):
            masks[index] = np.asarray(
                entry.cell.is_inside_cell(
                    px0,
                    py0,
                    pz0,
                    context=context,
                ),
                dtype=bool,
            )

        return masks

    def sample(self, px0, py0, pz0, mark_conflicts=False):
        """Return cell indices for each point, or -1 for background.

        With mark_conflicts=True, points claimed by more than one cell are
        returned as MULTI_DEFINED (-2). Otherwise the first matching cell is
        returned.
        """
        px0, py0, pz0 = np.broadcast_arrays(px0, py0, pz0)
        labels = np.full(px0.shape, -1, dtype=np.int32)
        if not self.entries:
            return labels

        masks = self.claim_masks(px0, py0, pz0)
        claimed = masks.any(axis=0)
        first = masks.argmax(axis=0)
        labels[claimed] = first[claimed]

        if mark_conflicts:
            overlaps = masks.sum(axis=0) > 1
            labels[overlaps] = MULTI_DEFINED

        return labels

    def find_cell(self, px0, py0, pz0):
        """Return the first matching cell index for a scalar point."""
        labels = self.sample(px0, py0, pz0)
        if labels.ndim != 0:
            raise ValueError("find_cell expects scalar coordinates")
        return int(labels.item())


@dataclass(frozen=True)
class OverlapReport:
    path: Path
    count: int
    max_claim_count: int


def write_overlap_report(
    geometry: Geometry,
    px0,
    py0,
    pz0,
    path,
    report_format="csv",
):
    """Write grid points claimed by multiple cells to CSV or JSON.

    The CSV file contains x, y, z, the number of claiming cells, and the
    corresponding cell IDs. JSON writes the same information as a list of
    point objects. Returns an OverlapReport summary.
    """
    if report_format not in ("csv", "json"):
        raise ValueError(f"unknown overlap report format: {report_format!r}")
    px0, py0, pz0 = np.broadcast_arrays(px0, py0, pz0)
    masks = geometry.claim_masks(px0, py0, pz0)
    counts = masks.sum(axis=0)
    overlap = counts > 1
    count = int(np.count_nonzero(overlap))
    max_claim_count = int(counts.max()) if counts.size else 0

    flat_overlap = np.flatnonzero(overlap)
    flat_masks = masks.reshape(len(geometry), -1)
    flat_x = np.asarray(px0).ravel()
    flat_y = np.asarray(py0).ravel()
    flat_z = np.asarray(pz0).ravel()
    cell_ids = geometry.cell_ids

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if report_format == "json":
        points = []
        for flat_index in flat_overlap:
            claimant_indices = np.flatnonzero(flat_masks[:, flat_index])
            points.append(
                {
                    "x": float(flat_x[flat_index]),
                    "y": float(flat_y[flat_index]),
                    "z": float(flat_z[flat_index]),
                    "claim_count": int(len(claimant_indices)),
                    "cell_ids": [int(cell_ids[index]) for index in claimant_indices],
                }
            )
        payload = {
            "overlap_count": count,
            "max_claim_count": max_claim_count,
            "points": points,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        with path.open("w", newline="", encoding="utf-8") as file:
            file.write("# CSG overlap warning\n")
            file.write(f"# overlapping_points: {count}\n")
            file.write("# grid_coordinates_are_in_physical_units\n")

            writer = csv.writer(file)
            writer.writerow(["x", "y", "z", "claim_count", "cell_ids"])

            for flat_index in flat_overlap:
                claimant_indices = np.flatnonzero(flat_masks[:, flat_index])
                claimant_ids = ";".join(str(cell_ids[index]) for index in claimant_indices)
                writer.writerow(
                    [
                        float(flat_x[flat_index]),
                        float(flat_y[flat_index]),
                        float(flat_z[flat_index]),
                        len(claimant_indices),
                        claimant_ids,
                    ]
                )

    return OverlapReport(
        path=path,
        count=count,
        max_claim_count=max_claim_count,
    )
