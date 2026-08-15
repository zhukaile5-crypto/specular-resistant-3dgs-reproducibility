#!/usr/bin/env python3
"""Verify paper figures, table values, and source snapshots against references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from tools.generate_tables import collect_all


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(".{}.partial".format(path.name))
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(partial), str(path))


def _record_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _read_checksums(path: Path) -> Dict[str, str]:
    checksums = {}
    if not path.is_file():
        return checksums
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest.lower()
    return checksums


def verify(package_root: Path, project_root: Path) -> Dict[str, Any]:
    manifest = json.loads((package_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    expected_values = json.loads((package_root / "reference/paper_expected_values.json").read_text(encoding="utf-8"))
    checks: List[Dict[str, Any]] = []

    _record_check(checks, "manifest figure count", len(manifest["figures"]) == 6, "found {}".format(len(manifest["figures"])))
    _record_check(checks, "manifest table count", len(manifest["tables"]) == 5, "found {}".format(len(manifest["tables"])))

    for item in manifest["figures"]:
        figure_id = int(item["id"])
        filename = item["paper_asset"]
        expected_hash = item["reference_sha256"].lower()
        live_final = project_root / "results/final_paper_figures" / filename
        live_ok = live_final.is_file() and sha256(live_final).lower() == expected_hash
        _record_check(checks, "live Figure {} hash".format(figure_id), live_ok, str(live_final))

        source_output = project_root / item["source_output"]
        source_ok = source_output.is_file() and sha256(source_output).lower() == expected_hash
        _record_check(checks, "Figure {} source-output hash".format(figure_id), source_ok, str(source_output))

    generated = collect_all(project_root)
    for key, expected_rows in expected_values.items():
        if key == "schema":
            continue
        actual_rows = generated.get(key)
        _record_check(
            checks,
            "{} displayed values".format(key),
            actual_rows == expected_rows,
            "{} rows compared".format(len(expected_rows)),
        )

    snapshot_checksums = _read_checksums(package_root / "SOURCE_SHA256SUMS.txt")
    _record_check(checks, "source checksum inventory present", bool(snapshot_checksums), "{} entries".format(len(snapshot_checksums)))
    for relative, expected_hash in sorted(snapshot_checksums.items()):
        path = package_root / relative
        actual_hash = sha256(path).lower() if path.is_file() else "missing"
        _record_check(checks, "source snapshot {}".format(relative), actual_hash == expected_hash, actual_hash)

    failures = [check for check in checks if not check["passed"]]
    return {
        "schema": "paper_reproducibility_verification_v1",
        "package_root": str(package_root),
        "project_root": str(project_root),
        "passed": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv or None)
    report = verify(args.package_root.resolve(), args.project_root.resolve())
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


if __name__ == "__main__":
    raise SystemExit(main())
