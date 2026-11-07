from abc import ABC, abstractmethod

import numpy as np

from .shapes import SignedShape


class CellExpression(ABC):
    """Internal boolean expression tree owned by a Cell."""

    @abstractmethod
    def _evaluate(self, px0, py0, pz0, context=None): ...

    def __and__(self, other):
        return Intersection([self, other])

    def __rand__(self, other):
        return Intersection([other, self])

    def __or__(self, other):
        return Union([self, other])

    def __ror__(self, other):
        return Union([other, self])

    def __invert__(self):
        return Complement(self)


class SurfaceTerm(CellExpression):
    def __init__(self, sign, surface):
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        self.sign = sign
        self.surface = surface

    def _evaluate(self, px0, py0, pz0, context=None):
        positive = np.asarray(
            self.surface.is_positive(px0, py0, pz0),
            dtype=bool,
        )
        return positive if self.sign == 1 else np.logical_not(positive)

    def __repr__(self):
        return f"SurfaceTerm({self.sign:+d}, {type(self.surface).__name__})"


class Intersection(CellExpression):
    def __init__(self, children):
        flattened = []
        for child in children:
            expression = as_expression(child)
            if isinstance(expression, Intersection):
                flattened.extend(expression.children)
            else:
                flattened.append(expression)
        self.children = tuple(flattened)

    def _evaluate(self, px0, py0, pz0, context=None):
        result = np.ones_like(np.asarray(px0), dtype=bool)
        for child in self.children:
            result = np.logical_and(
                result,
                child._evaluate(px0, py0, pz0, context),
            )
        return result

    def __repr__(self):
        return f"Intersection({list(self.children)!r})"


class Union(CellExpression):
    def __init__(self, children):
        flattened = []
        for child in children:
            expression = as_expression(child)
            if isinstance(expression, Union):
                flattened.extend(expression.children)
            else:
                flattened.append(expression)
        self.children = tuple(flattened)

    def _evaluate(self, px0, py0, pz0, context=None):
        result = np.zeros_like(np.asarray(px0), dtype=bool)
        for child in self.children:
            result = np.logical_or(
                result,
                child._evaluate(px0, py0, pz0, context),
            )
        return result

    def __repr__(self):
        return f"Union({list(self.children)!r})"


class Complement(CellExpression):
    def __init__(self, child):
        self.child = as_expression(child)

    def _evaluate(self, px0, py0, pz0, context=None):
        return np.logical_not(self.child._evaluate(px0, py0, pz0, context))

    def __repr__(self):
        return f"Complement({self.child!r})"


class CellReference(CellExpression):
    def __init__(self, cell_id):
        self.cell_id = cell_id

    def _evaluate(self, px0, py0, pz0, context=None):
        if context is None:
            raise ValueError("CellReference requires an EvaluationContext")
        return context.evaluate_cell(self.cell_id, px0, py0, pz0)

    def __repr__(self):
        return f"CellReference({self.cell_id})"


class EvaluationContext:
    def __init__(self, regions):
        self.regions = dict(regions)
        self.active = set()

    def evaluate_cell(self, cell_id, px0, py0, pz0):
        if cell_id in self.active:
            raise ValueError(f"circular cell reference detected: {cell_id}")

        try:
            region = self.regions[cell_id]
        except KeyError:
            raise ValueError(f"unknown cell id: {cell_id}") from None

        self.active.add(cell_id)
        try:
            if hasattr(region, "is_inside_cell"):
                return region.is_inside_cell(
                    px0,
                    py0,
                    pz0,
                    context=self,
                )
            return region._evaluate(px0, py0, pz0, self)
        finally:
            self.active.discard(cell_id)


def as_expression(value):
    if isinstance(value, CellExpression):
        return value
    if isinstance(value, SignedShape):
        return SurfaceTerm(value.sign, value.shape)
    return SurfaceTerm(1, value)


class Cell:
    """A CSG cell, corresponding to an MCNP/PHITS cell card."""

    def __init__(self, expression, transform=None):
        if isinstance(expression, CellExpression):
            self.expression = expression
        else:
            self.expression = Intersection(
                [
                    SurfaceTerm(item.sign, item.shape)
                    if isinstance(item, SignedShape)
                    else SurfaceTerm(1, item)
                    for item in expression
                ]
            )
        self.transform = transform

    def print_info(self):
        print(self.expression)

    def is_inside_cell(self, px0, py0, pz0, context=None):
        px0, py0, pz0 = np.broadcast_arrays(px0, py0, pz0)

        if self.transform is not None:
            px0, py0, pz0 = self.transform.global_to_local_xyz(
                px0,
                py0,
                pz0,
            )

        result = np.asarray(
            self.expression._evaluate(px0, py0, pz0, context),
            dtype=bool,
        )
        return result.item() if result.ndim == 0 else result

    def __repr__(self):
        if self.transform is None:
            return f"Cell({self.expression!r})"
        return f"Cell({self.expression!r}, transform={self.transform!r})"
