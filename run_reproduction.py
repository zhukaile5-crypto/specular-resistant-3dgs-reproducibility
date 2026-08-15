#!/usr/bin/env python3
"""Unified entry point for paper artifact inventory, verification, and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from tools.generate_tables import collect_all, render_outputs
from tools.verify_reference import verify, write_json_atomic


PACKAGE_ROOT = Path(__file__).resolve().parent


def default_project_root() -> Optional[Path]:
    """Return the enclosing repository when the package is in release_packages."""
    candidate = PACKAGE_ROOT.parents[1]
    if (candidate / "experiment_configs").is_dir() and (candidate / "results").is_dir():
        return candidate
    return None


def require_project_root(value: Optional[Path]) -> Path:
    root = value.resolve() if value is not None else default_project_root()
    if root is None:
        raise ValueError("The project root was not detected; pass --project-root explicitly")
    if not (root / "experiment_configs").is_dir():
        raise FileNotFoundError("Not a complete project root: {}".format(root))
    return root


def list_artifacts() -> None:
    manifest = json.loads((PACKAGE_ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    print("Figures")
    for item in manifest["figures"]:
        print(
            "  Figure {id}: {paper_asset} [{levels}]\n"
            "    {reproduction_note}".format(
                id=item["id"],
                paper_asset=item["paper_asset"],
                levels=", ".join(item["levels"]),
                reproduction_note=item["reproduction_note"],
            )
        )
    print("Tables")
    for item in manifest["tables"]:
        print(
            "  Table {id}: {label} [{levels}]\n"
            "    {limitation}".format(
                id=item["id"],
                label=item["label"],
                levels=", ".join(item["levels"]),
                limitation=item["limitation"],
            )
        )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all paper artifacts and reproduction levels")

    verify_parser = subparsers.add_parser("verify", help="Verify live results against frozen paper references")
    verify_parser.add_argument("--project-root", type=Path)
    verify_parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "verification_report.json")

    tables_parser = subparsers.add_parser("tables", help="Generate all five tables from live JSON/CSV results")
    tables_parser.add_argument("--project-root", type=Path)
    tables_parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "generated_tables")

    figures_parser = subparsers.add_parser("figures", help="Rebuild selected paper figures")
    figures_parser.add_argument("--project-root", type=Path)
    figures_parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "reproduced_figures")
    figures_parser.add_argument("--mode", choices=("compose", "recompute"), required=True)
    figures_parser.add_argument("--ids", nargs="+", default=["all"])

    args = parser.parse_args(argv or None)
    if args.command == "list":
        list_artifacts()
        return 0
    if args.command == "verify":
        report = verify(PACKAGE_ROOT, require_project_root(args.project_root))
        write_json_atomic(args.output.resolve(), report)
        print("Verification passed: {}".format(report["passed"]))
        print("Checks: {}, failures: {}".format(report["check_count"], report["failure_count"]))
        print(args.output.resolve())
        if not report["passed"]:
            for check in report["checks"]:
                if not check["passed"]:
                    print("FAILED: {} -- {}".format(check["name"], check["detail"]))
            return 1
        return 0
    if args.command == "tables":
        project_root = require_project_root(args.project_root)
        render_outputs(collect_all(project_root), args.output_dir.resolve())
        print("Generated Tables 1--5 in {}".format(args.output_dir.resolve()))
        return 0
    if args.command == "figures":
        # Figure dependencies are optional for inventory and table-only users.
        from tools.reproduce_figures import parse_ids, reproduce

        project_root = require_project_root(args.project_root)
        report = reproduce(PACKAGE_ROOT, project_root, args.output_dir.resolve(), parse_ids(args.ids), args.mode)
        for record in report["records"]:
            print(
                "Figure {figure}: byte_identical={byte_identical}, "
                "pixel_identical={pixel_identical} -> {output}".format(**record)
            )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
