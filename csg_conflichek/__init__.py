"""csg_conflichek: PHITS CSG overlap/undefined checker."""

from . import macro, shapes
from .cell import Cell, CellExpression
from .geometry import MULTI_DEFINED, CellEntry, Geometry, OverlapReport
from .phits_geom_syntax import load_phits_text
from .transforms import Transform

__version__ = "0.1.0"

__all__ = [
    "MULTI_DEFINED",
    "Cell",
    "CellEntry",
    "CellExpression",
    "Geometry",
    "OverlapReport",
    "Transform",
    "__version__",
    "load_phits_text",
    "macro",
    "shapes",
]
