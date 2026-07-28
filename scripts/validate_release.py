"""Validate release version bindings and offline asset hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse


EXPECTED_ASSETS = {"checkpoint.pth", "model.onnx"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_release(root: Path, tag: str) -> dict[str, object]:
    root = Path(root).resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(project.get("project", {}).get("version", ""))
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version:\s*[\"']?([^\s\"']+)", citation)
    citation_version = match.group(1) if match else ""
    manifest = json.loads((root / "configs/desktop_assets.json").read_text(encoding="utf-8"))
    manifest_version = str(manifest.get("release_version", ""))
    expected_tag = f"v{version}" if version else ""
    checks = {
        "semantic_version": bool(re.fullmatch(r"\d+\.\d+\.\d+", version)),
        "tag_matches_project": bool(tag) and tag == expected_tag,
        "citation_version": citation_version == version,
        "manifest_version": manifest_version == version,
        "asset_inventory": False,
        "asset_hashes": True,
        "asset_urls": True,
    }
    assets: list[dict[str, object]] = []
    rows = manifest.get("assets")
    names = [str(row.get("name")) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    checks["asset_inventory"] = (
        manifest.get("schema_version") == 1
        and isinstance(rows, list)
        and len(rows) == len(EXPECTED_ASSETS)
        and len(set(names)) == len(names)
        and set(names) == EXPECTED_ASSETS
    )
    for row in rows if isinstance(rows, list) else []:
        try:
            if not isinstance(row, dict):
                raise ValueError("asset entry must be an object")
            name = str(row["name"])
            path = (root / str(row["offline_path"])).resolve()
            if root not in path.parents or path.name != name:
                raise ValueError("asset path escapes repository")
            actual = _sha256(path)
            expected = str(row["sha256"]).lower()
            url = str(row.get("url", ""))
            parsed = urlparse(url)
            ok = bool(re.fullmatch(r"[0-9a-f]{64}", expected)) and actual == expected
            url_ok = (
                parsed.scheme == "https"
                and parsed.netloc == "github.com"
                and parsed.path == f"/quanntm1206/clearhop/releases/download/{expected_tag}/{name}"
                and not parsed.query
                and not parsed.fragment
            )
            assets.append({"name": row.get("name"), "path": str(path.relative_to(root)), "sha256": actual, "hash_match": ok, "url_version_match": url_ok})
            checks["asset_hashes"] &= ok
            checks["asset_urls"] &= url_ok
        except Exception as exc:
            checks["asset_hashes"] = False
            checks["asset_urls"] = False
            assets.append({"name": row.get("name") if isinstance(row, dict) else None, "error": f"{type(exc).__name__}: {exc}"})
    if not assets:
        checks["asset_inventory"] = False
        checks["asset_hashes"] = False
        checks["asset_urls"] = False
    status = "pass" if all(checks.values()) else "fail"
    return {"schema_version": 1, "status": status, "tag": tag, "version": version, "checks": checks, "assets": assets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_release(args.root, args.tag)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        output = args.output if args.output.is_absolute() else args.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
