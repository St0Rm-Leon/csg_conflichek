"""Cartesian bounding boxes for CSG surfaces and cells."""

from dataclasses import dataclass

import numpy as np

from . import macro, shapes as sp
from ._geom_syntax_core import TransformedShape
from .cell import CellReference, Complement, Intersection, SurfaceTerm, Union


class BoundsError(ValueError):
    """Raised when a finite bounding box cannot be determined."""


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self):
        if not (self.xmin < self.xmax and self.ymin < self.ymax and self.zmin < self.zmax):
            raise BoundsError("bounds must satisfy min < max on every axis")

    def expanded(self, margin):
        margin = float(margin)
        if margin < 0:
            raise BoundsError("margin must be non-negative")
        return Bounds(
            self.xmin - margin,
            self.xmax + margin,
            self.ymin - margin,
            self.ymax + margin,
            self.zmin - margin,
            self.zmax + margin,
        )


def _merge(bounds_list):
    if not bounds_list:
        return None
    return Bounds(
        min(item[0] for item in bounds_list),
        max(item[1] for item in bounds_list),
        min(item[2] for item in bounds_list),
        max(item[3] for item in bounds_list),
        min(item[4] for item in bounds_list),
        max(item[5] for item in bounds_list),
    )


def _points_bounds(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return None
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
        float(points[:, 2].min()),
        float(points[:, 2].max()),
    )


def _transformed_bounds(bounds, transform):
    if bounds is None or transform is None:
        return bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = np.array(
        [[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)],
        dtype=float,
    )
    return _points_bounds(transform.local_to_global(corners))


def _box_vertices(origin, vectors):
    points = [np.asarray(origin, dtype=float)]
    for vector in vectors:
        vector = np.asarray(vector, dtype=float)
        points = [point + sign * vector for point in points for sign in (0.0, 1.0)]
    return np.array(points, dtype=float)


def _sphere_bounds(shape):
    return (
        shape.x0 - shape.r,
        shape.x0 + shape.r,
        shape.y0 - shape.r,
        shape.y0 + shape.r,
        shape.z0 - shape.r,
        shape.z0 + shape.r,
    )


def _torus_bounds(shape, axis):
    radial = shape.A + shape.C
    if axis == 0:
        return (
            shape.x0 - shape.B,
            shape.x0 + shape.B,
            shape.y0 - radial,
            shape.y0 + radial,
            shape.z0 - radial,
            shape.z0 + radial,
        )
    if axis == 1:
        return (
            shape.x0 - radial,
            shape.x0 + radial,
            shape.y0 - shape.B,
            shape.y0 + shape.B,
            shape.z0 - radial,
            shape.z0 + radial,
        )
    return (
        shape.x0 - radial,
        shape.x0 + radial,
        shape.y0 - radial,
        shape.y0 + radial,
        shape.z0 - shape.B,
        shape.z0 + shape.B,
    )


def shape_bounds(shape):
    """Return (xmin,xmax,ymin,ymax,zmin,zmax) or None if unbounded."""

    if isinstance(shape, TransformedShape):
        return _transformed_bounds(shape_bounds(shape.shape), shape.transform)

    if isinstance(shape, sp.Sphere):
        return _sphere_bounds(shape)

    if isinstance(shape, sp.XTorus):
        return _torus_bounds(shape, 0)
    if isinstance(shape, sp.YTorus):
        return _torus_bounds(shape, 1)
    if isinstance(shape, sp.ZTorus):
        return _torus_bounds(shape, 2)

    if isinstance(shape, macro.RPP):
        return (shape.x0, shape.x1, shape.y0, shape.y1, shape.z0, shape.z1)

    if isinstance(shape, macro.BOX):
        vertices = _box_vertices(
            (shape.x0, shape.y0, shape.z0),
            [
                (shape.ax, shape.ay, shape.az),
                (shape.bx, shape.by, shape.bz),
                (shape.cx, shape.cy, shape.cz),
            ],
        )
        return _points_bounds(vertices)

    if isinstance(shape, macro.HEX):
        center = np.array([shape.x0, shape.y0, shape.z0], dtype=float)
        h = np.array([shape.hx, shape.hy, shape.hz], dtype=float)
        a = np.array([shape.ax, shape.ay, shape.az], dtype=float)
        b = np.array([shape.bx, shape.by, shape.bz], dtype=float)
        c = np.array([shape.cx, shape.cy, shape.cz], dtype=float)
        vertices = []
        for h_scale in (0.0, 1.0):
            for sa in (-1.0, 1.0):
                for sb in (-1.0, 1.0):
                    for sc in (-1.0, 1.0):
                        vertices.append(center + h_scale * h + sa * a + sb * b + sc * c)
        return _points_bounds(np.array(vertices, dtype=float))

    if isinstance(shape, macro.RCC):
        center = np.array([shape.x0, shape.y0, shape.z0], dtype=float)
        top = center + np.array([shape.Hx, shape.Hy, shape.Hz], dtype=float)
        vertices = np.array([center, top], dtype=float)
        points = np.vstack(
            [
                vertices - shape.R,
                vertices + shape.R,
            ]
        )
        return _points_bounds(points)

    if isinstance(shape, macro.REC):
        center = np.array([shape.x0, shape.y0, shape.z0], dtype=float)
        h = np.array([shape.hx, shape.hy, shape.hz], dtype=float)
        a = np.array([shape.ax, shape.ay, shape.az], dtype=float)
        if shape.is_b_vector:
            b = np.array([shape.bx, shape.by, shape.bz], dtype=float)
        else:
            b = np.zeros(3)
        vertices = []
        for h_scale in (0.0, 1.0):
            for sa in (-1.0, 1.0):
                for sb in (-1.0, 1.0):
                    point = center + h_scale * h + sa * a
                    if shape.is_b_vector:
                        point = point + sb * b
                    else:
                        point = point + np.array([shape.bx, shape.bx, shape.bx]) * sb
                    vertices.append(point)
        return _points_bounds(np.array(vertices, dtype=float))

    if isinstance(shape, macro.TRC):
        center = np.array([shape.x0, shape.y0, shape.z0], dtype=float)
        top = center + np.array([shape.hx, shape.hy, shape.hz], dtype=float)
        radius = max(shape.r1, shape.r2)
        points = np.vstack(
            [
                np.array([center, top]) - radius,
                np.array([center, top]) + radius,
            ]
        )
        return _points_bounds(points)

    if isinstance(shape, macro.ELL):
        center = np.array([shape.x0, shape.y0, shape.z0], dtype=float)
        axis = np.array([shape.ax, shape.ay, shape.az], dtype=float)
        half = np.sqrt((shape.a * axis) ** 2 + (shape.b**2) * (1.0 - axis**2))
        return _points_bounds(np.vstack([center - half, center + half]))

    if isinstance(shape, macro.ARB):
        return _points_bounds(np.array(shape.vertices, dtype=float))

    if isinstance(shape, macro.WED):
        center = np.array([shape.x0, shape.y0, shape.z0], dtype=float)
        h = np.array([shape.hx, shape.hy, shape.hz], dtype=float)
        a = np.array([shape.ax, shape.ay, shape.az], dtype=float)
        b = np.array([shape.bx, shape.by, shape.bz], dtype=float)
        vertices = []
        for h_scale in (0.0, 1.0):
            for offset in (np.zeros(3), a, b, a + b):
                vertices.append(center + h_scale * h + offset)
        return _points_bounds(np.array(vertices, dtype=float))

    return None


def iter_expression_surfaces(expression):
    if isinstance(expression, SurfaceTerm):
        yield expression.surface
    elif isinstance(expression, (Intersection, Union)):
        for child in expression.children:
            yield from iter_expression_surfaces(child)
    elif isinstance(expression, Complement):
        yield from iter_expression_surfaces(expression.child)
    elif isinstance(expression, CellReference):
        return


def estimate_bounds(model, margin=0.0):
    """Estimate bounds from finite surfaces and cell transforms."""

    pieces = []

    for surface in model.surfaces.values():
        bounds = shape_bounds(surface)
        if bounds is not None:
            pieces.append(bounds)

    for parsed in model.cells:
        transform = parsed.inline_transform
        if transform is None and parsed.transform_id:
            transform = model.transforms.get(parsed.transform_id)
        if transform is None:
            continue
        for surface in iter_expression_surfaces(parsed.expression):
            bounds = _transformed_bounds(shape_bounds(surface), transform)
            if bounds is not None:
                pieces.append(bounds)

    result = _merge(pieces)
    if result is None:
        raise BoundsError(
            "cannot determine a finite bounding box from the input surfaces; "
            "provide explicit bounds with --bounds"
        )

    return result.expanded(margin)
