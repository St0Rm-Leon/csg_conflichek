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


def _index_coordinates(flat_index, bounds, size):
    nx, ny, nz = size
    iz, remainder = divmod(int(flat_index), nx * ny)
    iy, ix = divmod(remainder, nx)
    x = bounds.xmin + ix * (bounds.xmax - bounds.xmin) / (nx - 1)
    y = bounds.ymin + iy * (bounds.ymax - bounds.ymin) / (ny - 1)
    z = bounds.zmin + iz * (bounds.zmax - bounds.zmin) / (nz - 1)
    return x, y, z


def _nearest_index(indices, bounds, size):
    nx, ny, nz = size
    ix = indices % nx
    iy = (indices // nx) % ny
    iz = indices // (nx * ny)
    mean_x = float(ix.mean())
    mean_y = float(iy.mean())
    mean_z = float(iz.mean())
    dx = (bounds.xmax - bounds.xmin) / (nx - 1)
    dy = (bounds.ymax - bounds.ymin) / (ny - 1)
    dz = (bounds.zmax - bounds.zmin) / (nz - 1)
    dist2 = ((ix - mean_x) * dx) ** 2 + ((iy - mean_y) * dy) ** 2 + ((iz - mean_z) * dz) ** 2
    tolerance = max(dx, dy, dz) ** 2 * 1e-12
    close = indices[dist2 <= float(dist2.min()) + tolerance]
    return int(close.min())


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
    sum_x = np.bincount(conflict_labels, weights=ix, minlength=count + 1)
    sum_y = np.bincount(conflict_labels, weights=iy, minlength=count + 1)
    sum_z = np.bincount(conflict_labels, weights=iz, minlength=count + 1)

    centroid_x = sum_x[1:] / sizes[1:]
    centroid_y = sum_y[1:] / sizes[1:]
    centroid_z = sum_z[1:] / sizes[1:]
    label_offset = conflict_labels - 1
    dx = (bounds.xmax - bounds.xmin) / (nx - 1)
    dy = (bounds.ymax - bounds.ymin) / (ny - 1)
    dz = (bounds.zmax - bounds.zmin) / (nz - 1)
    dist2 = (
        ((ix - centroid_x[label_offset]) * dx) ** 2
        + ((iy - centroid_y[label_offset]) * dy) ** 2
        + ((iz - centroid_z[label_offset]) * dz) ** 2
    )

    order = np.lexsort((indices, dist2, conflict_labels))
    sorted_labels = conflict_labels[order]
    sorted_dist = dist2[order]
    sorted_indices = indices[order]
    boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sorted_labels)]
    present_labels = sorted_labels[starts]
    tolerance = max(dx, dy, dz) ** 2 * 1e-12

    block_type = OVERLAP if cell_ids else UNDEFINED
    blocks = []
    for start, end, label in zip(starts, ends, present_labels):
        group_dist = sorted_dist[start:end]
        group_indices = sorted_indices[start:end]
        minimum = float(group_dist[0])
        best_index = int(group_indices[group_dist <= minimum + tolerance].min())
        x, y, z = _index_coordinates(best_index, bounds, size)
        blocks.append(
            ConflictBlock(
                type=block_type,
                x=x,
                y=y,
                z=z,
                cell_ids=cell_ids,
                point_count=int(sizes[label]),
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
        best_index = _nearest_index(index_array, bounds, size)
        x, y, z = _index_coordinates(best_index, bounds, size)
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

    blocks.sort(
        key=lambda block: (
            block.type != OVERLAP,
            block.x,
            block.y,
            block.z,
            block.cell_ids,
        )
    )
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


def _merge_group(points, bounds=None, size=None):
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

        if bounds is not None and size is not None:
            nx, ny, nz = size
            index_array = np.array(
                [(point.iz * ny + point.iy) * nx + point.ix for point in component],
                dtype=np.int64,
            )
            best_index = _nearest_index(index_array, bounds, size)
            x, y, z = _index_coordinates(best_index, bounds, size)
            block_type = component[0].type
            cell_ids = component[0].cell_ids
        else:
            mean_x = sum(point.x for point in component) / len(component)
            mean_y = sum(point.y for point in component) / len(component)
            mean_z = sum(point.z for point in component) / len(component)
            distances = [
                (point.x - mean_x) ** 2 + (point.y - mean_y) ** 2 + (point.z - mean_z) ** 2
                for point in component
            ]
            minimum = min(distances)
            tolerance = max(1.0, max(distances)) * 1e-12
            best = min(
                (
                    point
                    for point, distance in zip(component, distances)
                    if distance <= minimum + tolerance
                ),
                key=lambda point: (point.iz, point.iy, point.ix),
            )
            x, y, z = best.x, best.y, best.z
            block_type = best.type
            cell_ids = best.cell_ids

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


def merge_conflict_points(points, bounds=None, size=None):
    groups = {}
    for point in points:
        groups.setdefault(point.group, []).append(point)

    blocks = []
    for group_points in groups.values():
        blocks.extend(_merge_group(group_points, bounds, size))

    blocks.sort(
        key=lambda block: (
            block.type != OVERLAP,
            block.x,
            block.y,
            block.z,
            block.cell_ids,
        )
    )
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
    return merge_conflict_points(points, bounds, size)


@dataclass(frozen=True)
class StabilityReport:
    base_size: tuple[int, int, int]
    verify_size: tuple[int, int, int]
    base_overlap_points: int
    verify_overlap_points: int
    base_undefined_points: int
    verify_undefined_points: int
    base_overlap_blocks: int
    verify_overlap_blocks: int
    base_undefined_blocks: int
    verify_undefined_blocks: int
    base_overlap_volume: float
    verify_overlap_volume: float
    base_undefined_volume: float
    verify_undefined_volume: float


def _summary(blocks, bounds, size):
    nx, ny, nz = size
    cell_volume = (
        (bounds.xmax - bounds.xmin)
        / (nx - 1)
        * (bounds.ymax - bounds.ymin)
        / (ny - 1)
        * (bounds.zmax - bounds.zmin)
        / (nz - 1)
    )
    overlap_points = sum(block.point_count for block in blocks if block.type == OVERLAP)
    undefined_points = sum(block.point_count for block in blocks if block.type == UNDEFINED)
    return {
        "overlap_points": overlap_points,
        "undefined_points": undefined_points,
        "overlap_blocks": sum(1 for block in blocks if block.type == OVERLAP),
        "undefined_blocks": sum(1 for block in blocks if block.type == UNDEFINED),
        "overlap_volume": overlap_points * cell_volume,
        "undefined_volume": undefined_points * cell_volume,
    }


def compare_scans(model, bounds, size, verify_size, workers=0, base_blocks=None):
    size = tuple(int(value) for value in size)
    verify_size = tuple(int(value) for value in verify_size)
    if min(size) < 2 or min(verify_size) < 2:
        raise ValueError("grid size values must be >= 2")

    if base_blocks is None:
        base_blocks = scan_conflicts(model, bounds, size=size, workers=workers)
    verify_blocks = scan_conflicts(model, bounds, size=verify_size, workers=workers)
    base = _summary(base_blocks, bounds, size)
    verify = _summary(verify_blocks, bounds, verify_size)
    return StabilityReport(
        base_size=size,
        verify_size=verify_size,
        base_overlap_points=base["overlap_points"],
        verify_overlap_points=verify["overlap_points"],
        base_undefined_points=base["undefined_points"],
        verify_undefined_points=verify["undefined_points"],
        base_overlap_blocks=base["overlap_blocks"],
        verify_overlap_blocks=verify["overlap_blocks"],
        base_undefined_blocks=base["undefined_blocks"],
        verify_undefined_blocks=verify["undefined_blocks"],
        base_overlap_volume=base["overlap_volume"],
        verify_overlap_volume=verify["overlap_volume"],
        base_undefined_volume=base["undefined_volume"],
        verify_undefined_volume=verify["undefined_volume"],
    )


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
