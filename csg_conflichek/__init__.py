"""csg_conflichek: PHITS CSG overlap/undefined checker."""

from . import macro, shapes
from .cell import Cell, CellExpression
from .conflict import StabilityReport, compare_scans, scan_conflicts
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
    "StabilityReport",
    "Transform",
    "__version__",
    "compare_scans",
    "load_phits_text",
    "macro",
    "scan_conflicts",
    "shapes",
]
