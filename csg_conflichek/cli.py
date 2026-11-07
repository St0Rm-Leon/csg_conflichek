"""Command-line interface for the PHITS CSG conflict checker."""

import argparse
import re
import sys
from pathlib import Path

from . import __version__
from .bounds import Bounds, BoundsError, estimate_bounds
from .conflict import (
    compare_scans,
    count_by_type,
    scan_conflicts,
    write_conflicts_csv,
)
from .phits_geom_syntax import load_phits_text


def parse_size(value):
    if isinstance(value, bool):
        raise ValueError("size must be an integer or NX,NY,NZ")
    if isinstance(value, int):
        return (value, value, value)

    parts = [item for item in re.split(r"[x,\s]+", str(value).strip().lower()) if item]
    try:
        values = [int(item) for item in parts]
    except ValueError:
        raise ValueError("size must be an integer or NX,NY,NZ") from None

    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError("size must be an integer or NX,NY,NZ")
    if any(item < 2 for item in values):
        raise ValueError("size values must be >= 2")
    return tuple(values)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="csgc",
        description="Check PHITS CSG geometry for overlap and undefined regions.",
    )
    parser.add_argument("-i", "--input", required=True, help="PHITS input file")
    parser.add_argument(
        "-o",
        "--output",
        default="overlap.csv",
        help="output CSV path (default: overlap.csv)",
    )
    parser.add_argument(
        "--size",
        default="200",
        help="grid size: N or NX,NY,NZ (default: 200)",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="explicit Cartesian bounds",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="expand automatic bounds by this amount",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes for large grids (0 = auto, default: 0)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run a second scan at --verify-size and report stability",
    )
    parser.add_argument(
        "--verify-size",
        default=None,
        help="verification grid size (default: double the primary size)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="write every conflicting grid point instead of merged blocks",
    )
    parser.add_argument(
        "--fail-on-conflict",
        action="store_true",
        help="return exit code 1 when conflicts are found",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress summary output")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"csgc {__version__}",
    )
    return parser


def _relative_change(base, verify):
    if base == 0:
        return 0.0 if verify == 0 else float("inf")
    return (verify - base) / base


def _format_change(base, verify):
    if base == 0:
        return "n/a" if verify == 0 else "new"
    return f"{(verify - base) / base:+.2%}"


def _print_verification(report, quiet):
    warnings = []
    if abs(_relative_change(report.base_overlap_volume, report.verify_overlap_volume)) > 0.05:
        warnings.append("overlap volume changed by more than 5%")
    if abs(_relative_change(report.base_undefined_volume, report.verify_undefined_volume)) > 0.05:
        warnings.append("undefined volume changed by more than 5%")
    if report.base_overlap_blocks != report.verify_overlap_blocks:
        warnings.append("overlap block count changed")
    if report.base_undefined_blocks != report.verify_undefined_blocks:
        warnings.append("undefined block count changed")

    if not quiet:
        print(
            "verify: "
            f"{report.base_size[0]}x{report.base_size[1]}x{report.base_size[2]} -> "
            f"{report.verify_size[0]}x{report.verify_size[1]}x{report.verify_size[2]}"
        )
        print(
            "overlap volume: "
            f"{report.base_overlap_volume:.6g} -> {report.verify_overlap_volume:.6g} "
            f"({_format_change(report.base_overlap_volume, report.verify_overlap_volume)})"
        )
        print(
            "undefined volume: "
            f"{report.base_undefined_volume:.6g} -> {report.verify_undefined_volume:.6g} "
            f"({_format_change(report.base_undefined_volume, report.verify_undefined_volume)})"
        )
        print(f"overlap blocks: {report.base_overlap_blocks} -> {report.verify_overlap_blocks}")
        print(
            f"undefined blocks: {report.base_undefined_blocks} -> {report.verify_undefined_blocks}"
        )

    for warning in warnings:
        print(f"csgc: warning: {warning}", file=sys.stderr)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = None

    try:
        path = Path(args.input)
        if not path.is_file():
            raise OSError(f"input file not found: {args.input}")
        model = load_phits_text(path.read_text(encoding="utf-8"))

        if args.bounds is None:
            bounds = estimate_bounds(model, margin=args.margin)
        else:
            bounds = Bounds(*args.bounds)
            if args.margin:
                bounds = bounds.expanded(args.margin)

        size = parse_size(args.size)
        blocks = scan_conflicts(
            model,
            bounds,
            size=size,
            merge=not args.no_merge,
            workers=args.workers,
        )
        output = write_conflicts_csv(blocks, args.output)

        if args.verify:
            if args.verify_size is None:
                verify_size = tuple(value * 2 for value in size)
                if verify_size[0] * verify_size[1] * verify_size[2] > 200_000_000:
                    raise ValueError("verification grid is too large; provide --verify-size")
            else:
                verify_size = parse_size(args.verify_size)
            base_blocks = None if args.no_merge else blocks
            report = compare_scans(
                model,
                bounds,
                size,
                verify_size,
                workers=args.workers,
                base_blocks=base_blocks,
            )
    except (BoundsError, OSError, ValueError) as exc:
        print(f"csgc: error: {exc}", file=sys.stderr)
        return 1

    if report is not None:
        _print_verification(report, args.quiet)

    if not args.quiet:
        overlaps, undefined = count_by_type(blocks)
        print(
            f"There are {len(blocks)} conflicting locations in the geometry "
            f"provided by the input file {args.input}, details are written in the {output}."
        )
        print(f"overlap: {overlaps}, undefined: {undefined}")

    if args.fail_on_conflict and blocks:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
