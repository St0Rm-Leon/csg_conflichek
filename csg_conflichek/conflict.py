"""Grid-based overlap and undefined-region detection."""

import csv
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import generate_binary_structure, label as _nd_label
except ImportError:
    _nd_label = None
    _STRUCTURE = None
else:
    _STRUCTURE = generate_binary_structure(3, 1)


OVERLAP = "overlap"
UNDEFINED = "undefined"
_MAX_BBOX_VOLUME = 2_000_000
_DENSE_VOLUME_FACTOR = 8
_AUTO_WORKER_POINT_THRESHOLD = 20_000_000


@dataclass(frozen=True)
class ConflictPoint:
    ix: int
    iy: int
    iz: int
    x: float
    y: float
    z: float
    type: str
    cell_ids: tuple[int, ...] = ()

    @property
    def group(self):
        return (self.type, self.cell_ids)


@dataclass(frozen=True)
class ConflictBlock:
    type: str
    x: float
    y: float
    z: float
    cell_ids: tuple[int, ...] = ()
    point_count: int = 1


def _pack_codes(overlap_masks, n_cells):
    n_words = (n_cells + 63) // 64
    if n_words == 0:
        return np.empty((overlap_masks.shape[1], 0), dtype=np.uint64)

    words = []
    for word_index in range(n_words):
        chunk = overlap_masks[word_index * 64 : (word_index + 1) * 64].astype(np.uint64)
        bits = np.uint64(1) << np.arange(chunk.shape[0], dtype=np.uint64)
        words.append((chunk * bits[:, None]).sum(axis=0))

    if n_words == 1:
        return words[0]
    return np.stack(words, axis=1)


def _unique_codes(codes):
    if codes.ndim == 1:
        unique, inverse = np.unique(codes, return_inverse=True)
        return unique, inverse, [(int(code),) for code in unique]
    unique, inverse = np.unique(codes, axis=0, return_inverse=True)
    return unique, inverse, [tuple(int(value) for value in row) for row in unique]


def _collect_batches(model, bounds, size, z_start, z_stop):
    nx, ny, nz = size
    x = np.linspace(bounds.xmin, bounds.xmax, nx, dtype=float)
    y = np.linspace(bounds.ymin, bounds.ymax, ny, dtype=float)
    z = np.linspace(bounds.zmin, bounds.zmax, nz, dtype=float)

    n_cells = len(model.geometry)
    overlap_groups = {}
    undefined_indices = []

    for iz in range(z_start, z_stop):
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.full_like(x_grid, z[iz])
        masks = model.geometry.claim_masks(x_grid, y_grid, z_grid)
        counts = masks.sum(axis=0)
        base = iz * ny * nx

        undefined_flat = np.flatnonzero(counts == 0)
        if undefined_flat.size:
            undefined_indices.append(undefined_flat.astype(np.int64) + base)

        overlap_flat = np.flatnonzero(counts > 1)
        if not overlap_flat.size:
            continue

        iy, ix = np.divmod(overlap_flat, nx)
        overlap_masks = masks[:, iy, ix]
        codes = _pack_codes(overlap_masks, n_cells)
        _, inverse, keys = _unique_codes(codes)

        order = np.argsort(inverse, kind="stable")
        sorted_inverse = inverse[order]
        boundaries = np.flatnonzero(sorted_inverse[1:] != sorted_inverse[:-1]) + 1

        for group_positions, group_values in zip(
            np.split(order, boundaries),
            np.split(sorted_inverse, boundaries),
        ):
            key = keys[int(group_values[0])]
            indices = overlap_flat[group_positions].astype(np.int64) + base
            overlap_groups.setdefault(key, []).append(indices)

    return overlap_groups, undefined_indices


def _decode_cell_ids(key, cell_ids):
    result = []
    for word_index, word in enumerate(key):
        bits = int(word)
        while bits:
            lowest = bits & -bits
            bit = lowest.bit_length() - 1
            index = word_index * 64 + bit
            if index < len(cell_ids):
                result.append(cell_ids[index])
            bits ^= lowest
    return tuple(sorted(result))


def _worker_init(model, bounds, size):
    global _WORKER_STATE
    _WORKER_STATE = (model, bounds, size)


def _worker_scan(z_start, z_stop):
    model, bounds, size = _WORKER_STATE
    groups, undefined = _collect_batches(model, bounds, size, z_start, z_stop)
    compact_groups = {
        key: np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        for key, arrays in groups.items()
    }
    if undefined:
        compact_undefined = np.concatenate(undefined) if len(undefined) > 1 else undefined[0]
    else:
        compact_undefined = np.empty(0, dtype=np.int64)
    return compact_groups, compact_undefined


def _collect_batches_parallel(model, bounds, size, workers):
    nz = size[2]
    chunks = [chunk for chunk in np.array_split(np.arange(nz), workers) if chunk.size]
    ranges = [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(model, bounds, size),
    ) as executor:
        results = list(
            executor.map(
                _worker_scan,
                [item[0] for item in ranges],
                [item[1] for item in ranges],
            )
        )

    overlap_groups = {}
    undefined_indices = []
    for groups, undefined in results:
        for key, indices in groups.items():
            overlap_groups.setdefault(key, []).append(indices)
        if undefined.size:
            undefined_indices.append(undefined)
    return overlap_groups, undefined_indices


def _resolve_workers(workers, size):
    total_points = size[0] * size[1] * size[2]
    if workers is None or workers == 0:
        if total_points < _AUTO_WORKER_POINT_THRESHOLD:
            workers = 1
        else:
            workers = min(os.cpu_count() or 1, 8)

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return min(workers, size[2])


def _mean_coordinates(indices, bounds, size):
    nx, ny, _ = size
    ix = indices % nx
    iy = (indices // nx) % ny
    iz = indices // (ny * nx)

    x = bounds.xmin + float(ix.mean()) * (bounds.xmax - bounds.xmin) / (nx - 1)
    y = bounds.ymin + float(iy.mean()) * (bounds.ymax - bounds.ymin) / (ny - 1)
    z = bounds.zmin + float(iz.mean()) * (bounds.zmax - bounds.zmin) / (size[2] - 1)
    return x, y, z


def _merge_indices_dense(indices, cell_ids, bounds, size, ix, iy, iz):
    nx, ny, nz = size
    x0, x1 = int(ix.min()), int(ix.max())
    y0, y1 = int(iy.min()), int(iy.max())
    z0, z1 = int(iz.min()), int(iz.max())
    sx = x1 - x0 + 1
    sy = y1 - y0 + 1
    sz = z1 - z0 + 1

    local = ((iz - z0) * sy + (iy - y0)) * sx + (ix - x0)
    mask = np.zeros(sx * sy * sz, dtype=bool)
    mask[local] = True
    labels, count = _nd_label(mask.reshape(sz, sy, sx), structure=_STRUCTURE)
    if count == 0:
        return []

    conflict_labels = labels.ravel()[local]
    sizes = np.bincount(conflict_labels, minlength=count + 1)
    sum_x = np.bincount(conflict_labels, weights=ix - x0, minlength=count + 1)
    sum_y = np.bincount(conflict_labels, weights=iy - y0, minlength=count + 1)
    sum_z = np.bincount(conflict_labels, weights=iz - z0, minlength=count + 1)

    block_type = OVERLAP if cell_ids else UNDEFINED
    blocks = []
    for label in range(1, count + 1):
        point_count = int(sizes[label])
        if point_count == 0:
            continue
        mean_x = x0 + float(sum_x[label]) / point_count
        mean_y = y0 + float(sum_y[label]) / point_count
        mean_z = z0 + float(sum_z[label]) / point_count
        blocks.append(
            ConflictBlock(
                type=block_type,
                x=bounds.xmin + mean_x * (bounds.xmax - bounds.xmin) / (nx - 1),
                y=bounds.ymin + mean_y * (bounds.ymax - bounds.ymin) / (ny - 1),
                z=bounds.zmin + mean_z * (bounds.zmax - bounds.zmin) / (nz - 1),
                cell_ids=cell_ids,
                point_count=point_count,
            )
        )
    return blocks


def _merge_indices_sparse(indices, cell_ids, bounds, size):
    nx, ny, nz = size
    stride_y = nx
    stride_z = nx * ny
    remaining = set(int(value) for value in indices)
    block_type = OVERLAP if cell_ids else UNDEFINED
    blocks = []

    while remaining:
        start = remaining.pop()
        component = [start]
        stack = [start]

        while stack:
            current = stack.pop()
            iz, remainder = divmod(current, stride_z)
            iy, ix = divmod(remainder, stride_y)
            neighbors = []
            if ix > 0:
                neighbors.append(current - 1)
            if ix + 1 < nx:
                neighbors.append(current + 1)
            if iy > 0:
                neighbors.append(current - stride_y)
            if iy + 1 < ny:
                neighbors.append(current + stride_y)
            if iz > 0:
                neighbors.append(current - stride_z)
            if iz + 1 < nz:
                neighbors.append(current + stride_z)

            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)

        index_array = np.array(component, dtype=np.int64)
        x, y, z = _mean_coordinates(index_array, bounds, size)
        blocks.append(
            ConflictBlock(
                type=block_type,
                x=x,
                y=y,
                z=z,
                cell_ids=cell_ids,
                point_count=len(component),
            )
        )

    return blocks


def _merge_indices(indices, cell_ids, bounds, size):
    if _nd_label is not None:
        nx, ny, _ = size
        ix = indices % nx
        iy = (indices // nx) % ny
        iz = indices // (nx * ny)
        volume = (
            (int(ix.max()) - int(ix.min()) + 1)
            * (int(iy.max()) - int(iy.min()) + 1)
            * (int(iz.max()) - int(iz.min()) + 1)
        )
        limit = max(_MAX_BBOX_VOLUME, indices.size * _DENSE_VOLUME_FACTOR)
        if volume <= limit:
            return _merge_indices_dense(indices, cell_ids, bounds, size, ix, iy, iz)
    return _merge_indices_sparse(indices, cell_ids, bounds, size)


def _merge_batches(overlap_groups, undefined_indices, bounds, size, cell_ids):
    blocks = []
    for key, batches in overlap_groups.items():
        indices = np.concatenate(batches) if len(batches) > 1 else batches[0]
        ids = _decode_cell_ids(key, cell_ids)
        blocks.extend(_merge_indices(indices, ids, bounds, size))

    if undefined_indices:
        indices = (
            np.concatenate(undefined_indices)
            if len(undefined_indices) > 1
            else undefined_indices[0]
        )
        blocks.extend(_merge_indices(indices, (), bounds, size))

    blocks.sort(key=lambda block: (block.type != OVERLAP, block.x, block.y, block.z))
    return blocks


def _batches_to_points(model, bounds, size, batches):
    overlap_groups, undefined_indices = batches
    nx, ny, nz = size
    x_axis = np.linspace(bounds.xmin, bounds.xmax, nx, dtype=float)
    y_axis = np.linspace(bounds.ymin, bounds.ymax, ny, dtype=float)
    z_axis = np.linspace(bounds.zmin, bounds.zmax, nz, dtype=float)
    cell_ids = model.geometry.cell_ids
    points = []

    for key, arrays in overlap_groups.items():
        ids = _decode_cell_ids(key, cell_ids)
        indices = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        for flat_index in indices:
            iz, remainder = divmod(int(flat_index), nx * ny)
            iy, ix = divmod(remainder, nx)
            points.append(
                ConflictPoint(
                    ix=ix,
                    iy=iy,
                    iz=iz,
                    x=float(x_axis[ix]),
                    y=float(y_axis[iy]),
                    z=float(z_axis[iz]),
                    type=OVERLAP,
                    cell_ids=ids,
                )
            )

    for indices in undefined_indices:
        for flat_index in indices:
            iz, remainder = divmod(int(flat_index), nx * ny)
            iy, ix = divmod(remainder, nx)
            points.append(
                ConflictPoint(
                    ix=ix,
                    iy=iy,
                    iz=iz,
                    x=float(x_axis[ix]),
                    y=float(y_axis[iy]),
                    z=float(z_axis[iz]),
                    type=UNDEFINED,
                    cell_ids=(),
                )
            )

    return points


def find_conflict_points(model, bounds, size):
    size = tuple(int(value) for value in size)
    if size[0] < 2 or size[1] < 2 or size[2] < 2:
        raise ValueError("grid size values must be >= 2")
    batches = _collect_batches(model, bounds, size, 0, size[2])
    return _batches_to_points(model, bounds, size, batches)


def _mean(values):
    return sum(values) / len(values)


def _merge_group(points):
    by_index = {(point.ix, point.iy, point.iz): point for point in points}
    remaining = set(by_index)
    blocks = []

    while remaining:
        start = remaining.pop()
        component = [by_index[start]]
        queue = deque([start])

        while queue:
            ix, iy, iz = queue.popleft()
            for neighbor in (
                (ix - 1, iy, iz),
                (ix + 1, iy, iz),
                (ix, iy - 1, iz),
                (ix, iy + 1, iz),
                (ix, iy, iz - 1),
                (ix, iy, iz + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(by_index[neighbor])
                    queue.append(neighbor)

        first = component[0]
        blocks.append(
            ConflictBlock(
                type=first.type,
                x=_mean([point.x for point in component]),
                y=_mean([point.y for point in component]),
                z=_mean([point.z for point in component]),
                cell_ids=first.cell_ids,
                point_count=len(component),
            )
        )

    return blocks


def merge_conflict_points(points):
    groups = {}
    for point in points:
        groups.setdefault(point.group, []).append(point)

    blocks = []
    for group_points in groups.values():
        blocks.extend(_merge_group(group_points))

    blocks.sort(key=lambda block: (block.type != OVERLAP, block.x, block.y, block.z))
    return blocks


def scan_conflicts(model, bounds, size=(200, 200, 200), merge=True, workers=0):
    size = tuple(int(value) for value in size)
    if size[0] < 2 or size[1] < 2 or size[2] < 2:
        raise ValueError("grid size values must be >= 2")

    workers = _resolve_workers(workers, size)
    if workers <= 1:
        batches = _collect_batches(model, bounds, size, 0, size[2])
    else:
        batches = _collect_batches_parallel(model, bounds, size, workers)

    if not merge:
        points = _batches_to_points(model, bounds, size, batches)
        return [
            ConflictBlock(
                type=point.type,
                x=point.x,
                y=point.y,
                z=point.z,
                cell_ids=point.cell_ids,
                point_count=1,
            )
            for point in points
        ]

    if _nd_label is not None:
        return _merge_batches(
            batches[0],
            batches[1],
            bounds,
            size,
            model.geometry.cell_ids,
        )

    points = _batches_to_points(model, bounds, size, batches)
    return merge_conflict_points(points)


def _cell_columns(block):
    if block.type == UNDEFINED:
        return "", ""
    if len(block.cell_ids) == 1:
        return str(block.cell_ids[0]), ""
    if len(block.cell_ids) == 2:
        return str(block.cell_ids[0]), str(block.cell_ids[1])
    return ";".join(str(cell_id) for cell_id in block.cell_ids), ""


def write_conflicts_csv(blocks, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["no", "type", "x", "y", "z", "cell1", "cell2"])
        for number, block in enumerate(blocks, start=1):
            cell1, cell2 = _cell_columns(block)
            writer.writerow(
                [
                    number,
                    block.type,
                    _format_number(block.x),
                    _format_number(block.y),
                    _format_number(block.z),
                    cell1,
                    cell2,
                ]
            )

    return path


def _format_number(value):
    text = f"{value:.12g}"
    return "0" if text == "-0" else text


def count_by_type(blocks):
    overlaps = sum(1 for block in blocks if block.type == OVERLAP)
    undefined = sum(1 for block in blocks if block.type == UNDEFINED)
    return overlaps, undefined
